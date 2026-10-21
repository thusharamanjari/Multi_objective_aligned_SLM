#!/usr/bin/env python
# coding: utf-8

# ### Unsloth


import torch
from datasets import load_dataset
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import pandas as pd
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from peft import LoraConfig, get_peft_model
from peft import PeftModel


# MULTI
from accelerate import PartialState#MULTI
from transformers import TrainingArguments

device_map='auto' # for DP and running with `python test_sft.py`
device_map="DDP" # for DDP and running with `accelerate launch test_sft.py`

if device_map == "DDP":
    device_string = PartialState().process_index
    device_map={'':device_string}
# MULTI --end


import evaluate
print('modules loaded')


max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.


################----- #MULTI-GPU training----


# Load the model + tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    cache_dir='',
    use_cache = False,
    device_map = device_map,
)

# PEFT config
lora_alpha = 8
lora_dropout = 0.1
lora_r = 32
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
)


# (Optional) sanity check
print("Active adapter:", getattr(model, "active_adapter", None))


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!


print('lora loaded')


# <a name="Data"></a>
# ### Data Prep
# We now use the Alpaca dataset from [yahma](https://huggingface.co/datasets/yahma/alpaca-cleaned), which is a filtered version of 52K of the original [Alpaca dataset](https://crfm.stanford.edu/2023/03/13/alpaca.html). You can replace this code section with your own data prep.
# 
# **[NOTE]** To train only on completions (ignoring the user's input) read TRL's docs [here](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only).
# 
# **[NOTE]** Remember to add the **EOS_TOKEN** to the tokenized output!! Otherwise you'll get infinite generations!
# 
# If you want to use the `llama-3` template for ShareGPT datasets, try our conversational [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Alpaca.ipynb)
# 
# For text completions like novel writing, try this [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb).


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass


####gpt
train_data = load_dataset("json", data_files="models_in/FT_data_CCD_detect_train_629_shuffled_GPT.json")
val_data = load_dataset("json", data_files="models_in/FT_data_CCD_detect_val_136_shuffled_GPT.json")
test_data = load_dataset("json", data_files="models_in/FT_data_CCD_detect_test_136_shuffled_GPT.json")


train_data= train_data['train']
val_data= val_data['train']
test_data= test_data['train']


train_data = train_data.map(formatting_prompts_func, batched = True,)
val_data = val_data.map(formatting_prompts_func, batched = True,)
test_data = test_data.map(formatting_prompts_func, batched = True,)


print(' train_data: ',len(train_data))
print(' val_data: ',len(val_data))
print(' test_data: ',len(test_data))


train_data[0]


test_data[0]


val_data[0]


# <a name="Train"></a>
# ### Train the model
# Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support TRL's `DPOTrainer`!


def formatting_func(example):
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:\n"
    )
    return {"prompt": prompt, "response": example["output"]}


from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback
# 629/ 2*4= 78 steps per epochs
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    peft_config=peft_config,
    train_dataset=train_data,
    eval_dataset=val_data,
    formatting_func=formatting_func,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,

        # evaluate & save every step so early stopping can act quickly
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,

        logging_steps=1,
        max_steps=234,                    # hard cap; training may stop earlier
        learning_rate=1e-4,
        fp16=False,                       # or bf16=True
        bf16=True,
        warmup_ratio=0.1,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="/N/slate/tnaduvil/models_save/MULTI/CD_SFT",
        report_to="none",

        # --- Best-model restore + metric config ---
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters = False

    ),
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,     # stop if no improvement for 3 evals
        )
    ],
)


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


#save trainer stats to a csv
import pandas as pd

# Step 1: Extract logs
log_history = trainer.state.log_history  # list of dicts

# Step 2: Convert to DataFrame
df_logs = pd.DataFrame(log_history)

# Step 3: Save to CSV
df_logs.to_csv("models_out/MULTI/CD_SFT_trainer_stats.csv", index=False)

print("Training Loss	and Validation Loss saved to trainer_stats.csv")
