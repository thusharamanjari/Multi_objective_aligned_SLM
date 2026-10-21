#!/usr/bin/env python
# coding: utf-8

# ### Unsloth


from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",  # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",  # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


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


from datasets import load_dataset


#================================data from CCD block========


#load train val test data
import pandas as pd
from pathlib import Path

base = Path("models_in")

test_data_label  = pd.read_csv(base / "dial_test_data_label_GPT.csv")


print(len(test_data_label))

t = 'models_out/CD_generated_vs_reference_F1.csv'
CCdetect_results = pd.read_csv(t)


CC_detected = CCdetect_results['Generated'].to_list()
# Convert each string into list of sentences
processed = []
for val in CC_detected:
    if pd.isna(val):  # keep NaN values as is
        processed.append([])
    else:
        # Split on '.' and strip spaces, drop empties
        parts = [s.strip() for s in val.split('.') if s.strip()]
        processed.append(parts)

test_data_label['CCD_feature'] = processed
import pandas as pd
import json


# Load your CSV file
dialogues_data = test_data_label


# Assuming your dataframe is already defined as `dialogues_data`
instruction_ft_data = []

for _, row in dialogues_data.iterrows():
    entry = {
        "instruction": (
            "You are simulating a patient with substance use disorder. Based on the patient's history and "
            "the counselor's question, generate a realistic patient response that expresses the given cognitive feature."
        ),
        "input": json.dumps({
            "patient_history": row["patient_history"],
            "counselor_question": row["couns_question"],
            "cognitive_feature_name": row["CCD_feature_name"],
            "cognitive_feature_value": row["CCD_feature"]
        }, ensure_ascii=False),
        "output": row["dialogue"]
    }
    instruction_ft_data.append(entry)

# Save as .jsonl
with open("models_in/instruction_ft_dialogues_data_test_v2_GPT.json", "w", encoding="utf-8") as f:
    json.dump(instruction_ft_data, f, indent=4)

print("✅ instruction_ft_data_v1.jsonl successfully created.")


#====================


train_data = load_dataset("json", data_files="models_in/instruction_ft_dialogues_data_train_v2_GPT.json")
val_data = load_dataset("json", data_files="models_in/instruction_ft_dialogues_data_val_v2_GPT.json")
test_data = load_dataset("json", data_files="models_in/instruction_ft_dialogues_data_test_v2_GPT.json")


train_data= train_data['train']
val_data= val_data['train']
test_data= test_data['train']


#save test data
import pandas as pd
import json
# Parse each test_data instance
parsed_entries = []

for instance in test_data:
    input_data = json.loads(instance["input"])
    parsed_entries.append({
        "patient_history": input_data["patient_history"],
        "counselor_question": input_data["counselor_question"],
        "CCD_feature_name": input_data["cognitive_feature_name"],
        "CCD_feature_value": input_data["cognitive_feature_value"]
    })

# Create DataFrame
test_df = pd.DataFrame(parsed_entries)


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
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=val_data,
    formatting_func=formatting_func,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,

        # evaluate & save every step so early stopping can act quickly
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,

        logging_steps=1,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=False,                       # or bf16=True
        bf16=True,
        warmup_ratio=0.1,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="/N/slate/tnaduvil/models_save/Dial_SFT",
        report_to="none",

        # --- Best-model restore + metric config ---
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=6,     # stop if no improvement for 3 evals
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
df_logs.to_csv("/N/slate/tnaduvil/models_out/dial_SFT_trainer_stats.csv", index=False)

print("Trainer statistics saved to trainer_stats.csv")


model.save_pretrained("models_out/Dial_SFT/lora_model_SFT_dial_gpt")  # Local saving
tokenizer.save_pretrained("models_out/Dial_SFT/lora_model_SFT_dial_gpt")


# ### baseline results
