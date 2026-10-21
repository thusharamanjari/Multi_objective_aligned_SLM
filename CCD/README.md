# CCD

CCD detection using supervised fine-tuning (SFT) and direct preference optimization (DPO).


## Detailed inputs and outputs


### `1_1_SFT.py`

| Type | File or directory | Role |
| --- | --- | --- |
| Input | `models_in/FT_data_CCD_detect_train_629_shuffled_GPT.json` | Training data |
| Input | `models_in/FT_data_CCD_detect_val_136_shuffled_GPT.json` | Validation data |
| Input | `models_in/FT_data_CCD_detect_test_136_shuffled_GPT.json` | Test data |
| Model input | `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face base model and tokenizer |
| Output directory | `/N/slate/tnaduvil/models_save/MULTI/CD_SFT/` | Trainer checkpoints, under `checkpoint-<step>/` |
| Output file | `models_out/MULTI/CD_SFT_trainer_stats.csv` | Training and evaluation logs |


### `1_2_DPO.py`

| Type | File or directory | Role |
| --- | --- | --- |
| Input | `models_in/dpo_pairs_cc_detect.csv` | Preference data with `prompt`, `chosen`, and `rejected` columns |
| Model input directory | `/N/slate/tnaduvil/models_save/CD_SFT/checkpoint-100/` | SFT adapter loaded for DPO |
| Model input | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | Hugging Face base model |
| Output directory | `/N/slate/tnaduvil/models_save/CD_DPO/` | Trainer checkpoints, under `checkpoint-<step>/` |
| Output file | `models_out/CD_DPO_trainer_stats.csv` | Training and evaluation logs |


Environment definitions are at the repository root: [`dpo_env.yml`](../dpo_env.yml) and [`grpo_env.yml`](../grpo_env.yml).
