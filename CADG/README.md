# CADG

CADG dialogue generation using supervised fine-tuning (SFT), direct preference optimization (DPO), and group relative policy optimization (GRPO).


## Detailed inputs and outputs

### `2_1_SFT_cadg.py`

| Type | File or directory | Role |
| --- | --- | --- |
| Input | `models_in/dial_test_data_label_GPT.csv` | Labeled test dialogues; uses `patient_history`, `couns_question`, `CCD_feature_name`, and `dialogue` |
| Input | `models_out/CD_generated_vs_reference_F1.csv` | CCD detection results; reads the `Generated` column |
| Input | `models_in/instruction_ft_dialogues_data_train_v2_GPT.json` | Training data |
| Input | `models_in/instruction_ft_dialogues_data_val_v2_GPT.json` | Validation data |
| Output, then input | `models_in/instruction_ft_dialogues_data_test_v2_GPT.json` | Test JSON generated from the two CSV inputs, then loaded as a dataset |
| Model input | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | Hugging Face base model and tokenizer |
| Output directory | `/N/slate/tnaduvil/models_save/Dial_SFT/` | Trainer checkpoints, under `checkpoint-<step>/` |
| Output file | `/N/slate/tnaduvil/models_out/dial_SFT_trainer_stats.csv` | Training and evaluation logs |
| Output directory | `models_out/Dial_SFT/lora_model_SFT_dial_gpt/` | Final adapter and tokenizer saved with `save_pretrained` |



### `2_2_dpo_cadg.py`

| Type | File or directory | Role |
| --- | --- | --- |
| Input | `models_in/dpo_pairs_dial_selective.csv` | Preference data with `prompt`, `chosen`, and `rejected` columns |
| Model input directory | `/N/slate/tnaduvil/models_save/Dial_SFT/checkpoint-112/` | SFT adapter loaded for DPO |
| Model input | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | Hugging Face base model |
| Output directory | `/N/slate/tnaduvil/models_save/Dial_DPO/` | Trainer checkpoints, under `checkpoint-<step>/` |
| Output file | `models_out/DPO_dial_trainer_stats.csv` | Training and evaluation logs |
| Output directory | `models_save/dial_DPO_lora_model/` | Final adapter and tokenizer saved with `save_pretrained`; used by GRPO |

### `2_3_grpo_cadg.py`

| Type | File or directory | Role |
| --- | --- | --- |
| Input | `models_in/instruction_ft_dialogues_data_train_v2_GPT.json` | Training data |
| Input | `models_in/instruction_ft_dialogues_data_val_v2_GPT.json` | Validation data loaded and prepared by the script |
| Input | `models_in/instruction_ft_dialogues_data_test_v2_GPT.json` | Test data |
| Model input directory | `models_save/dial_DPO_lora_model/` | DPO adapter and preferred tokenizer; tokenizer falls back to the base model if loading fails |
| Model input | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | Hugging Face base model |
| Output directory | `/N/slate/tnaduvil/models_save/Dial_grpo_new/` | Trainer output directory; checkpoint creation depends on trainer save settings and run length |
| Output, then input | `training_generations_v_1_new.csv` | Incremental prompt, completion, and reward recorder; read back after training |
| Output file | `grpo_progress_table_new.csv` | Progress table written at training end when log rows exist |
| Output file | `models_out/GRPO_training_generations_grpo_attention_new.csv` | Attention reward records |
| Output file | `models_out/GRPO_training_generations_grpo_length_new.csv` | Length reward records |
| Output file | `models_out/GRPO_trainer_stats_grpo_attn_len_90_new.csv` | Trainer logs |
| Output directory | `models_save/Dial_grpo/dial_grpo_lora_model_grpo/` | Final adapter and tokenizer saved with `save_pretrained` |
| Output file | `models_out/GRPO_test_inference_results_grpo_attn_len_90_new.csv` | Test inference results |


Environment definitions are at the repository root: [`dpo_env.yml`](../dpo_env.yml) and [`grpo_env.yml`](../grpo_env.yml).
