#!/usr/bin/env python
# coding: utf-8

# ### Unsloth


# One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer

PatchDPOTrainer()


from unsloth import FastLanguageModel
from peft import PeftModel

base = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  # or your real base
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base, max_seq_length=4096, dtype=None, load_in_4bit=True
)

# Now point to your adapter folder copied from Colab
model = PeftModel.from_pretrained(model, "/N/slate/tnaduvil/models_save/Dial_SFT/checkpoint-112", is_trainable=True)  # your adapter dir

#outputs_SFT_CD_GPT


# @title Alignment Handbook utils
import os
import re
from typing import List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"] = "sft",
    assistant_prefix="<|assistant|>\n",
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            prompt_messages = [
                [msg for msg in example["chosen"] if msg["role"] == "user"][0]
            ]
            # Insert system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])
            # TODO: handle case where chosen/rejected also have system messages
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = _strip_prefix(
                example["text_chosen"], assistant_prefix
            )
            example["text_rejected"] = _strip_prefix(
                example["text_rejected"], assistant_prefix
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def get_datasets(
    data_config: dict,
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    if type(data_config) is dict:
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    return raw_datasets


def mix_datasets(
    dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, split=split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(
                    f"Split type {split} not recognized as one of test or train."
                )

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(
                seed=42
            )
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets


# <a name="Data"></a>
# ### Data Prep
# We follow Huggingface's [Alignment Handbook](https://github.com/huggingface/alignment-handbook) for [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) and use the [Ultra Feedback dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), and sample 0.5% of it to speed things up. You can sample the full dataset for a full run.


# We shall print a random item from the dataset


import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# --- Inputs -------------------------------------------------
# 1) Your pandas DF with columns: 'prompt', 'chosen', 'rejected'
#    Each is a plain string.
dpo_df = pd.read_csv('models_in/dpo_pairs_dial_selective.csv')#oct_in/dpo_pairs_dial_selective.csv

# 2) Your tokenizer (must support .apply_chat_template)

# If your tokenizer doesn't have a chat template, fall back to your DEFAULT_CHAT_TEMPLATE
try:
    _ = tokenizer.apply_chat_template([{"role":"user", "content":"test"}], tokenize=False)
except Exception:
    # Use the DEFAULT_CHAT_TEMPLATE you pasted earlier
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE


# --- Step 1: Make a HF Dataset from the DF ------------------
base_ds = Dataset.from_pandas(
    dpo_df[["prompt", "chosen", "rejected"]].copy(),  # ensure only needed cols
    preserve_index=False
)

# --- Step 2: Convert each row into the "messages" structure expected by apply_chat_template(task='dpo')
def to_messages(ex):
    prompt = str(ex["prompt"]) if ex["prompt"] is not None else ""
    chosen = str(ex["chosen"]) if ex["chosen"] is not None else ""
    rejected = str(ex["rejected"]) if ex["rejected"] is not None else ""
    return {
        # For DPO in the utils you pasted:
        # - 'chosen' and 'rejected' each start with a USER message (prompt),
        #   followed by ASSISTANT message (completion).
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
    }

ds_msgs = base_ds.map(
    to_messages,
    remove_columns=["prompt", "chosen", "rejected"],
    desc="Wrapping rows into chat message lists",
)

# --- Step 3: Apply chat template for DPO to get text_prompt / text_chosen / text_rejected
# Uses your apply_chat_template() defined in the Alignment Handbook utils.
ds_fmt = ds_msgs.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer, "task": "dpo", "assistant_prefix": "<|assistant|>\n"},
    num_proc=min(8, os.cpu_count() or 1),
    desc="Formatting comparisons with prompt template",
)

# --- Step 4: Rename to TRL’s expected columns and drop extras
ds_fmt = ds_fmt.rename_columns(
    {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
)

# Keep only the three fields TRL expects
keep = {"prompt", "chosen", "rejected"}
drop_cols = [c for c in ds_fmt.column_names if c not in keep]
if drop_cols:
    ds_fmt = ds_fmt.remove_columns(drop_cols)


# --- Optional: quick sanity check
row = ds_fmt[0]
print("PROMPT:\n", row["prompt"][:500], "...\n")
print("CHOSEN:\n", row["chosen"][:300], "...\n")
print("REJECTED:\n", row["rejected"][:300], "...\n")


import pandas as pd
from datasets import Dataset, DatasetDict
raw_dataset = ds_fmt

# --- 1) Identify the last occurrence positions for each column name
cols = raw_dataset.column_names  # e.g., ['chosen','rejected','chosen','rejected','prompt']
last_pos = {
    "prompt": max(i for i, c in enumerate(cols) if c == "prompt"),
    "chosen": max(i for i, c in enumerate(cols) if c == "chosen"),
    "rejected": max(i for i, c in enumerate(cols) if c == "rejected"),
}

# --- 2) Round-trip to pandas to pick columns by position (disambiguates duplicates)
pdf = raw_dataset.to_pandas()  # may show duplicate names; we use positions

clean_pdf = pd.DataFrame({
    "prompt":  pdf.iloc[:, last_pos["prompt"]],
    "chosen":  pdf.iloc[:, last_pos["chosen"]],
    "rejected": pdf.iloc[:, last_pos["rejected"]],
})

# (Optional) quick sanity check: if earlier duplicates differ, warn
def _diff_count(a, b):
    return (a.fillna("").astype(str).values != b.fillna("").astype(str).values).sum()

if cols.count("chosen") > 1:
    first_pos = min(i for i, c in enumerate(cols) if c == "chosen")
    diffs = _diff_count(pdf.iloc[:, first_pos], pdf.iloc[:, last_pos["chosen"]])
    if diffs:
        print(f"[WARN] 'chosen' duplicates differ in {diffs} rows; kept the LAST copy.")
if cols.count("rejected") > 1:
    first_pos = min(i for i, c in enumerate(cols) if c == "rejected")
    diffs = _diff_count(pdf.iloc[:, first_pos], pdf.iloc[:, last_pos["rejected"]])
    if diffs:
        print(f"[WARN] 'rejected' duplicates differ in {diffs} rows; kept the LAST copy.")

# --- 3) Back to a clean HF Dataset (no index column)
clean_ds = Dataset.from_pandas(clean_pdf, preserve_index=False)
print(clean_ds)
# -> Dataset({ features: ['prompt','chosen','rejected'], num_rows: 2120 })

# --- 4) Train/test split
clean_ds = clean_ds.shuffle(seed=42)
raw_datasets = clean_ds.train_test_split(test_size=0.10, seed=42)
raw_datasets = DatasetDict({"train": raw_datasets["train"], "test": raw_datasets["test"]})

print(raw_datasets)


import pprint

row = raw_datasets["train"][8]
pprint.pprint(row["prompt"])
pprint.pprint(row["chosen"])
pprint.pprint(row["rejected"])


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!


# <a name="Train"></a>
# ### Train the DPO model
# Now let's train our model. We do 3 epochs on 0.5% of the dataset to speed things up.


# One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer

PatchDPOTrainer()


from transformers import EarlyStoppingCallback
from trl import DPOTrainer, DPOConfig

dpo_args = DPOConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    num_train_epochs=3,
    learning_rate=5e-6,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.0,
    lr_scheduler_type="linear",
    seed=42,
    output_dir="/N/slate/tnaduvil/models_save/Dial_DPO",
    report_to="none",

    # ---- eval/save like SFT ----
    eval_strategy="steps",          # alias of evaluation_strategy
    eval_steps=1,                   # evaluate every step (like your SFT example)
    save_strategy="steps",
    save_steps=1,
    save_total_limit=2,

    # ---- best-model + metric ----
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # ---- DPO-specific hygiene ----
    remove_unused_columns=False,    # keep prompt/chosen/rejected
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,                 # TRL will handle a reference copy if None
    args=dpo_args,
    beta=0.1,
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["test"],
    tokenizer=tokenizer,
    max_length=1024,
    max_prompt_length=512,
)

# Train (will evaluate, early-stop, and reload best checkpoint automatically)
dpo_trainer.train()


#save trainer stats to a csv
import pandas as pd

# Step 1: Extract logs
log_history = dpo_trainer.state.log_history  # list of dicts

# Step 2: Convert to DataFrame
df_logs = pd.DataFrame(log_history)

# Step 3: Save to CSV
df_logs.to_csv("models_out/DPO_dial_trainer_stats.csv", index=False)

print("Training Loss	and Validation Loss saved to trainer_stats.csv")


model.save_pretrained("models_save/dial_DPO_lora_model")  # Local saving
tokenizer.save_pretrained("models_save/dial_DPO_lora_model")
