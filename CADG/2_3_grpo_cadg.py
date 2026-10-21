

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


#############################################load patient data--------------------------
from datasets import load_dataset

train_data = load_dataset("json", data_files="models_in/instruction_ft_dialogues_data_train_v2_GPT.json")
val_data = load_dataset("json", data_files="models_in/instruction_ft_dialogues_data_val_v2_GPT.json")
test_data = load_dataset("json", data_files="models_in/instruction_ft_dialogues_data_test_v2_GPT.json")

train_data= train_data['train']
val_data= val_data['train']
test_data= test_data['train']


train_data[0]


#===============================patch begin


import re, ast, json

def _strip_matching_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s

def _flatten_strings(x):
    if isinstance(x, (list, tuple)):
        out = []
        for el in x:
            out.extend(_flatten_strings(el))
        return out
    elif isinstance(x, str):
        return [x]
    else:
        return [str(x)]

def _to_clean_string(raw_val: str) -> str:
    if raw_val is None:
        return "NaN"

    s = str(raw_val).strip()
    # drop trailing comma if present before closing brace
    s = re.sub(r',\s*$', '', s)

    # remove one layer of matching quotes if present
    s = _strip_matching_quotes(s)

    # unescape backslash-escaped quotes that appear from JSON embedding
    s = s.replace('\\"', '"').replace("\\'", "'").strip()

    # quick checks for empties / NaN-ish
    compact = re.sub(r"\s+", "", s).lower()
    if compact in ("nan", "none", "null", "", "[]", "[[]]", "[[[]]]"):
        return "NaN"

    # try JSON array first (e.g., ["a","b"])
    parsed_list = None
    try:
        maybe = json.loads(s)
        if isinstance(maybe, list):
            parsed_list = maybe
        elif isinstance(maybe, str):
            s = maybe  # plain string wrapped as JSON
    except Exception:
        pass

    # try Python literal list (e.g., ['a', 'b'])
    if parsed_list is None and s.startswith("[") and s.endswith("]"):
        try:
            maybe = ast.literal_eval(s)
            if isinstance(maybe, list):
                parsed_list = maybe
            elif isinstance(maybe, str):
                s = maybe
        except Exception:
            # fall back to string-based cleanup below
            pass

    if parsed_list is not None:
        items = [t.strip().strip('"').strip("'") for t in _flatten_strings(parsed_list)]
        items = [i for i in items if i and i.lower() not in ("nan", "none", "null")]
        return ", ".join(items) if items else "NaN"

    # not a list -> treat as plain string; remove square brackets if any
    s = s.replace("[", "").replace("]", "").strip()
    if re.sub(r"\s+", "", s).lower() in ("nan", "none", "null", ""):
        return "NaN"
    return s

def replace_cf_value(text: str) -> str:
    """
    Replace the value of "cognitive_feature_value" with a cleaned single-quoted string.
    Keeps the rest of the JSON-like text unchanged.
    """
    pat = re.compile(r'("cognitive_feature_value"\s*:\s*)(.*?)(\s*})', re.DOTALL)

    def repl(m):
        raw_val = m.group(2)
        cleaned = _to_clean_string(raw_val)
        # write back in single quotes to match your requested format
        return f'{m.group(1)}\'{cleaned}\'{m.group(3)}'

    # only replace the first occurrence (it's the last field)
    return pat.sub(repl, text, count=1)

# --- Apply to your HF Datasets column ---
# Overwrite train_data["input"] with cleaned strings
train_data = train_data.map(lambda row: {"input": replace_cf_value(row["input"])})
test_data = test_data.map(lambda row: {"input": replace_cf_value(row["input"])})
val_data = val_data.map(lambda row: {"input": replace_cf_value(row["input"])})
#


#================================patch


from datasets import DatasetDict, Features, Value


def to_prompt_completion(ds):
    def _convert(example):
        inst = (example.get("instruction") or "").strip()
        inp  = (example.get("input") or "").strip()
        # Combine instruction + input (omit input if empty)
        prompt = f"{inst}\n\n{inp}" if inp else inst
        return {
            "prompt": prompt,
            "completion": (example.get("output") or "").strip(),
        }

    # Map to new schema and drop the original columns
    new_ds = ds.map(_convert, remove_columns=ds.column_names)
    # Ensure consistent schema
    return new_ds.cast(Features({"prompt": Value("string"),
                                 "completion": Value("string")}))

dataset = DatasetDict({
    "train": to_prompt_completion(train_data),
    "validation": to_prompt_completion(val_data),
    "test": to_prompt_completion(test_data),
})


from datasets import DatasetDict

def render_chat_row(example):
    p = example["prompt"]
    if isinstance(p, list) and p and isinstance(p[0], dict):
        # render with the model's chat template
        rendered = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        return {"prompt": rendered}          # overwrite with string
    return {"prompt": p}                     # already a string

dataset = DatasetDict({
    split: dataset[split].map(render_chat_row, desc=f"Render {split} to text")
    for split in dataset.keys()
})

# sanity check
ex = dataset["train"][0]["prompt"]
assert isinstance(ex, str), type(ex)


from datasets import DatasetDict

def to_ultrafeedback(example):
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "completion": example["completion"],
    }

dataset = DatasetDict({
    split: dataset[split].map(
        to_ultrafeedback,
        remove_columns=dataset[split].column_names,
        desc=f"Converting {split}",
    )
    for split in dataset.keys()
})

dataset_train = dataset["train"].remove_columns(
    [c for c in dataset["train"].column_names if c != "prompt"]
)


dataset_val = dataset["validation"].remove_columns(
    [c for c in dataset["validation"].column_names if c != "prompt"]
)

dataset_test = dataset["test"].remove_columns(
    [c for c in dataset["test"].column_names if c != "prompt"]
)

dataset_test_label = dataset["test"].remove_columns(
    [c for c in dataset["test"].column_names if c != "completion"]
)
print(dataset_test)


#############################################load patient data end--------------------------
token_hf = 'your hugging face token'

from huggingface_hub import login
login(token=token_hf)  # paste your token

print(dataset)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

adapter_path = "models_save/dial_DPO_lora_model" 
model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
print("model_id:", model_id)

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 1) Load base (4-bit) and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="eager",  # needed if you want attentions later
    quantization_config=quant_cfg,
)
# Prefer the tokenizer saved with your LoRA (keeps any added special tokens)
try:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# 2) Attach LoRA adapter weights
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

# (Optional) sanity check
print("Active adapter:", getattr(model, "active_adapter", None))


print(model.print_trainable_parameters())


# ## attention reward CC aligned


import os, re, ast, torch
import pandas as pd
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from IPython.display import display, clear_output


# JSON span helpers + raw prompt builder
# ==========================
def json_string_value_span(s, key):
    """
    Returns (start, end) char span of the JSON string value for "key": " ... ",
    within the given string s (which is your prompt text). Handles escaped quotes.
    """
    pat = rf'"{re.escape(key)}"\s*:\s*"(?P<val>(?:[^"\\]|\\.)*)"'
    m = re.search(pat, s, flags=re.DOTALL)
    return None if not m else (m.start("val"), m.end("val"))

def json_array_value_span(s, key):
    """
    Returns (start, end) char span of the JSON array value for "key": [ ... ],
    tolerant of quoted strings inside the array. Does not support nested arrays.
    """
    pat = rf'"{re.escape(key)}"\s*:\s*\[(?P<val>(?:[^\[\]]|"(?:(?:[^"\\]|\\.)*)")*)\]'
    m = re.search(pat, s, flags=re.DOTALL)
    return None if not m else (m.start("val"), m.end("val"))

def _extract_user_prompt_content(messages):
    """
    Given list[dict] chat messages, prefer the last 'user' message's content.
    Fallback to first message with 'content'.
    """
    if isinstance(messages, list):
        user_msgs = [m.get("content") for m in messages
                     if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)]
        if user_msgs:
            return user_msgs[-1]
        for m in messages:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                return m["content"]
    return None

def build_raw_from_example(ex):
    """
    Normalizes a variety of input shapes into:
      {"prompt": <string containing your JSON>, "messages": <list[chat dicts]>}
    """
    # String → single-user message
    if isinstance(ex, str):
        return {"prompt": ex, "messages": [{"role": "user", "content": ex}]}

    # Dict with direct prompt
    if isinstance(ex, dict):
        if ex.get("prompt"):
            p = ex["prompt"]
            return {"prompt": p, "messages": [{"role": "user", "content": p}]}
        if ex.get("instruction") and ex.get("input"):
            p = f"{ex['instruction']}\n\n{ex['input']}"
            return {"prompt": p, "messages": [{"role": "user", "content": p}]}

    # list[dict] chat messages
    if isinstance(ex, list) and ex and isinstance(ex[0], dict) and "role" in ex[0]:
        user_content = _extract_user_prompt_content(ex)
        if user_content:
            return {"prompt": user_content, "messages": ex}

    return None

def _extract_completion_text(completion_obj):
    """
    Accepts string, dict{'content'}, or list[dict] chat messages.
    Returns assistant text string.
    """
    if isinstance(completion_obj, str):
        return completion_obj
    if isinstance(completion_obj, dict):
        if isinstance(completion_obj.get("content"), str):
            return completion_obj["content"]
        return str(completion_obj)
    if isinstance(completion_obj, list) and completion_obj:
        for m in completion_obj:
            if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                return m["content"]
        for m in completion_obj:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                return m["content"]
        return str(completion_obj[0])
    return str(completion_obj)

#============================update


def json_single_string_value_span(s, key):
    """
    Returns (start, end) span of a single-quoted JSON-like string value:
      "key": '...'
    Handles escaped quotes like \'
    """
    pat = rf'"{re.escape(key)}"\s*:\s*\'(?P<val>(?:[^\'\\]|\\.)*)\''
    m = re.search(pat, s, flags=re.DOTALL)
    return None if not m else (m.start("val"), m.end("val"))


#============================update END

# ==========================

#==================begin
def _extract_section_spans(prompt, rendered, base):
    json_start = prompt.find("{")
    task_span_in_prompt = (0, json_start if json_start != -1 else len(prompt))
    context_span_in_prompt = json_string_value_span(prompt, "patient_history")

    # NEW: detect all shapes for cognitive_feature_value
    feat_name_str    = json_string_value_span(prompt, "cognitive_feature_name")
    feat_value_str_d = json_string_value_span(prompt, "cognitive_feature_value")      # "value"
    feat_value_str_s = json_single_string_value_span(prompt, "cognitive_feature_value")# 'value'
    feat_value_arr   = json_array_value_span(prompt,  "cognitive_feature_value")       # [ ... ]

    # Build feature spans with preference: array > "string" > 'string' > (optional) bareword
    feature_spans_in_prompt = []
    if feat_value_arr:
        feature_spans_in_prompt.append(feat_value_arr)
    elif feat_value_str_d:
        feature_spans_in_prompt.append(feat_value_str_d)
    elif feat_value_str_s:
        feature_spans_in_prompt.append(feat_value_str_s)

    question_span_in_prompt = json_string_value_span(prompt, "counselor_question")

    # --- DEBUG (optional) ---
    def _slice(s, span): return s[span[0]:span[1]] if span else None
    feature_txts = [_slice(prompt, s) for s in feature_spans_in_prompt if s]
    # --- end DEBUG ---

    def shift(span): return None if span is None else (span[0] + base, span[1] + base)
    return (
        shift(task_span_in_prompt),
        shift(context_span_in_prompt),
        [shift(s) for s in feature_spans_in_prompt if s],
        shift(question_span_in_prompt),
    )
#========================end


def _token_indices_for_sections(text_for_tokenizer, rendered_prompt, prompt, base):
    """
    Tokenize combined_text and map tokens to your sections.
    Assumes global tokenizer/model.
    """
    enc = tokenizer(
        text_for_tokenizer,
        return_tensors="pt",
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        add_special_tokens=True,
    )
    input_ids      = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    offsets        = enc["offset_mapping"][0].tolist()
    special_mask   = enc["special_tokens_mask"][0].tolist()

    task_span, context_span, feature_spans, question_span = _extract_section_spans(prompt, rendered_prompt, base)
    prompt_span_rendered = (base, base + len(prompt))

    def overlaps(tok_span, sec_span):
        if sec_span is None: return False
        a,b = tok_span; c,d = sec_span
        return (a < d) and (c < b)
    def in_prompt(tok_span):
        a,b = tok_span; c,d = prompt_span_rendered
        return (a < d) and (c < b)

    task_idx, ctx_idx, feat_idx, ques_idx, prompt_idx, gen_idx = [], [], [], [], [], []
    for i, (st, en) in enumerate(offsets):
        if special_mask[i] == 1:
            continue
        tok_span = (st, en)
        if in_prompt(tok_span):
            prompt_idx.append(i)
            if overlaps(tok_span, task_span): task_idx.append(i)
            if overlaps(tok_span, context_span): ctx_idx.append(i)
            if any(overlaps(tok_span, fs) for fs in feature_spans): feat_idx.append(i)
            if overlaps(tok_span, question_span): ques_idx.append(i)
        else:
            gen_idx.append(i)  # treat as generated token
    return input_ids, attention_mask, task_idx, ctx_idx, feat_idx, ques_idx, prompt_idx, gen_idx


# ==========================
# Attention-based reward for one (prompt, completion)
# ==========================
def attnCalc_pair(prompt_obj, completion_text):
    raw = build_raw_from_example(prompt_obj)
    if raw is None:
        return 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0

    prompt   = raw["prompt"]                          # user string containing JSON
    messages = raw.get("messages", [{"role":"user","content":prompt}])

    # Render prompt side (no assistant header), so generated region starts exactly at completion
    use_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
    if use_chat:
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        base = rendered_prompt.find(prompt)
        if base < 0:
            rendered_prompt = prompt
            base = 0
    else:
        rendered_prompt = prompt
        base = 0

    completion_text = _extract_completion_text(completion_text)
    combined_text = rendered_prompt + (completion_text or "")

    # Map tokens & sections
    (input_ids, attention_mask, task_idx, ctx_idx, feat_idx, ques_idx,
     prompt_idx, gen_idx) = _token_indices_for_sections(combined_text, rendered_prompt, prompt, base)

    if len(gen_idx) == 0 or len(prompt_idx) == 0:
        return 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0

    # Forward with attentions
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, use_cache=False)
    if was_training:
        model.train()

    # last layer attention: [H, T, T] → mean over heads → [T, T]
    last_attn  = out.attentions[-1][0]
    heads_mean = last_attn.mean(dim=0)

    # Average attention from generated tokens (queries) to all source tokens
    q = torch.tensor(gen_idx, device=heads_mean.device)
    attn_from_gen = heads_mean[q].mean(dim=0)  # [T]

    # Normalize mass over the union of section tokens
    union = sorted(set(task_idx + ctx_idx + feat_idx + ques_idx))
    keep = torch.zeros_like(attn_from_gen, dtype=torch.bool)
    if union:
        keep[torch.tensor(union, device=attn_from_gen.device)] = True
    Z = (attn_from_gen[keep]).sum().clamp_min(1e-12) if union else torch.tensor(1.0, device=attn_from_gen.device)

    def part_mass(idxs):
        if not idxs: return 0.0
        idx = torch.tensor(idxs, device=attn_from_gen.device)
        return float(attn_from_gen[idx].sum() / Z)

    s_task = part_mass(task_idx)
    s_ctx  = part_mass(ctx_idx)
    s_feat = part_mass(feat_idx)
    s_ques = part_mass(ques_idx)

    def density(norm_mass, n_tokens): return float(norm_mass / n_tokens) if n_tokens > 0 else 0.0
    d_task = density(s_task, len(task_idx))
    d_ctx  = density(s_ctx,  len(ctx_idx))
    d_feat = density(s_feat, len(feat_idx))
    d_ques = density(s_ques, len(ques_idx))

    # Reward = feature mass + beta * max(0, feature density - mean(other densities)), scaled
    def grpo_reward_optionA(m_task, m_ctx, m_feat, m_ques, d_task, d_ctx, d_feat, d_ques, beta=0.1):
        d_other = (d_task + d_ctx + d_ques) / 3.0
        return float(m_feat + beta * max(0.0, d_feat - d_other))
    reward = grpo_reward_optionA(s_task, s_ctx, s_feat, s_ques, d_task, d_ctx, d_feat, d_ques, beta=0.1) * 100.0


    #==========================
    # --- weights / strengths ---
    w_f = 0.6    # weight on feature mass
    w_q = 0.4    # weight on question mass (w_f + w_q must be 1)
    b   = 0.8    # density bonus strength
    c   = 0.6    # synergy strength (GM term)
    K   = 0.90  # scale factor

    # Averages excluding the target
    d_bar_not_feat = (d_task + d_ctx + d_ques) / 3.0
    d_bar_not_ques = (d_task + d_ctx + d_feat) / 3.0

    # Base mass (weighted arithmetic mean)
    base = w_f * s_feat + w_q * s_ques

    # Density bonuses (both targets)
    bonusF = max(0.0, d_feat - d_bar_not_feat)
    bonusQ = max(0.0, d_ques - d_bar_not_ques)
    bonus  = (b / 2.0) * (bonusF + bonusQ)

    # Synergy via weighted geometric mean (safe with tiny epsilon)
    eps = 1e-12
    synergy_gm = c * ((max(s_feat, eps) ** w_f) * (max(s_ques, eps) ** w_q))

    # Final reward
    reward = K * (base + bonus + synergy_gm)

    #===========================
    print('ATTENTION REWARD=========>>>::::::::', reward)

    return d_task, d_ctx, d_feat, d_ques, s_task, s_ctx, s_feat, s_ques, reward


#===================CPY REWARD -----start

import re
import ast

# ==========================
# Durable recorder (CSV)
# ==========================
# ------------------ Durable recorder: writes incrementally to CSV ------------------
class GenRecorder:
    """
    Append (prompt, completion, reward) rows during training.
    Writes incrementally to CSV so you don't lose data if training stops. #data_out/training_generations_v_1.csv
    """
    def __init__(self, path="training_generations_v_1_new.csv", flush_every=1):
        self.path = path
        self.flush_every = flush_every
        self.buffer = []
        self.header_written = os.path.exists(path) and os.path.getsize(path) > 0

    @staticmethod
    def _prompt_text(p):
        raw = build_raw_from_example(p)
        return raw["prompt"] if raw else (p if isinstance(p, str) else str(p))

    def append_many(self, prompts, completions, rewards):
        for p, c, r in zip(prompts, completions, rewards):
            self.buffer.append({"prompt": self._prompt_text(p), "completion": c, "reward": float(r)})
            if len(self.buffer) >= self.flush_every:
                self.flush()

    def flush(self):
        if not self.buffer: 
            return
        df = pd.DataFrame(self.buffer, columns=["prompt", "completion", "reward"])
        mode = "a" if self.header_written else "w"
        header = not self.header_written
        df.to_csv(self.path, mode=mode, header=header, index=False)
        self.header_written = True
        self.buffer.clear()

    def load_df(self):
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            return pd.read_csv(self.path)
        return pd.DataFrame(columns=["prompt", "completion", "reward"])

# global recorder you can reuse
recorder = GenRecorder(path="training_generations_v_1_new.csv", flush_every=1)

# Global recorder (change path if you like)


# ==========================
# Reward wrapper (vectorized over lists)
# ==========================
# Global collectors
prompts_list = []
completions_list = []
rewards_list = []

def attnReward(prompts, completions, **kwargs):
    rewards = []
    flat_completions_text = []
    for p, c in zip(prompts, completions):
        c_text = _extract_completion_text(c)
        flat_completions_text.append(c_text)
        *_, reward = attnCalc_pair(p, c_text)
        rewards.append(reward)

        # Debug prints (trim to 500 chars for sanity)
        raw = build_raw_from_example(p)
        dbg_prompt = raw["prompt"] if raw else (p if isinstance(p, str) else str(p))
        print("prompts: ", (dbg_prompt or "")[:500])
        print("completion: ", (c_text or "")[:500])
        print("$$$$$$$$$$$$$$$$$$$$")
        print("################+++++++++++++++##############")
        # Append to global collectors
        prompts_list.append(dbg_prompt)
        completions_list.append(c_text)
        rewards_list.append(reward)

    recorder.append_many(prompts, completions, rewards)


    return rewards


#==========================cpy reward
import re
import ast


# ==========================
# Helpers (completion -> str)
# ==========================
def _as_text(x):
    """Return plain string from a completion that may be str / dict / list-of-dicts."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        v = x.get("content")
        return v if isinstance(v, str) else str(x)
    if isinstance(x, list) and x:
        # prefer assistant message; else first item with 'content'
        for m in x:
            if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                return m["content"]
        for m in x:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                return m["content"]
        return str(x[0])
    return str(x)


# =============== Text / n-gram utils ===============
def normalize(s: str) -> str:
    """Lowercase, keep word chars/whitespace/apostrophes, collapse spaces."""
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    s = normalize(s)
    return s.split() if s else []

def ngram_set(tok_list, k=3):
    n = len(tok_list)
    if n < k:
        return set()
    return {tuple(tok_list[i:i+k]) for i in range(n - k + 1)}

def trigram_overlap(gen_str: str, base_str: str) -> float:
    """
    Overlap = |T_gen ∩ T_base| / max(1, |T_base|)
    """
    Tg = ngram_set(tokens(gen_str), 3)
    Tb = ngram_set(tokens(base_str), 3)
    denom = max(1, len(Tb))
    return len(Tg & Tb) / float(denom)

def copy_signed_reward_trigram(overlap3: float, tau=0.6, lam=1.0, gamma=2.0) -> float:
    """
    Positive if overlap <= tau (linear from 1 at 0 → 0 at tau),
    Negative if overlap > tau (penalty curve parameterized by lam, gamma).
    """
    # Clamp just in case of numeric drift
    overlap = max(0.0, min(1.0, float(overlap3)))


    tau = float(tau)

    # Handle edge cases
    if tau <= 0.0:
        # Everything is in the "negative" branch except zero
        x = overlap  # normalized to [0,1]
        lin = -x
        cur = -(x ** max(1e-8, gamma))
        return (1 - lam) * lin + lam * cur
    if tau >= 1.0:
        # Everything is in the "positive" branch
        z = overlap  # normalized to [0,1]
        lin = 1 - z
        cur = 1 - (z ** max(1e-8, gamma))
        return (1 - lam) * lin + lam * cur

    if overlap <= tau:
        # Positive side: +1 → 0
        z = overlap / tau  # in [0,1]
        lin = 1.0 - z
        cur = 1.0 - (z ** max(1e-8, gamma))
        return (1 - lam) * lin + lam * cur
    else:
        # Negative side: 0 → -1
        x = (overlap - tau) / (1.0 - tau)  # in [0,1]
        lin = -x
        cur = -(x ** max(1e-8, gamma))
        return (1 - lam) * lin + lam * cur

# =============== Prompt parsing helpers ===============
def _find_balanced_brace_block(s: str, start_idx: int):
    """
    Find substring from the first '{' at/after start_idx up to its matching '}'.
    Returns (substring_including_braces, end_index_after_block) or (None, None).
    """
    depth = 0
    begin = None
    for i in range(start_idx, len(s)):
        ch = s[i]
        if ch == '{':
            depth += 1
            if depth == 1:
                begin = i
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and begin is not None:
                    return s[begin:i+1], i+1
    return None, None

def json_any_string_value(s: str, key: str):
    """
    Return the unquoted text for "key": '...' OR "key": "..." (handles escaped quotes).
    """
    pat = rf'"{re.escape(key)}"\s*:\s*(?P<q>[\'"])(?P<val>(?:\\.|(?!\1).)*)\1'
    m = re.search(pat, s, flags=re.DOTALL)
    if not m:
        return None
    val = m.group("val")
    # Unescape common backslash-escaped quotes
    val = val.replace(r"\'", "'").replace(r'\"', '"')
    return val

def _pull_content_from_prompt_item(prompt_item) -> str:
    """
    Accepts either dict {'content': ..., 'role': ...} or list of such dicts.
    Returns the first content string found; else stringifies the object.
    """
    if isinstance(prompt_item, dict):
        return prompt_item.get("content", "") or ""
    if isinstance(prompt_item, list) and prompt_item:
        for x in prompt_item:
            if isinstance(x, dict) and "content" in x:
                return x["content"] or ""
    return str(prompt_item or "")

def extract_field_string_from_prompt(prompt_item, key: str, parse_list_like_if_string: bool = True) -> str:
    """
    Extract and normalize a string field (e.g., 'counselor_question' or 'cognitive_feature_value')
    from the prompt 'content'. Robust to single/double quotes and hybrid JSON/Python blocks.

    If the extracted string looks like a Python list (e.g., "['a','b']") and
    parse_list_like_if_string=True, it will be parsed and joined with spaces.
    """
    # 1) get content
    content = _pull_content_from_prompt_item(prompt_item)
    if not content:
        return ""

    # 2) find the dict block (prefer blank line + '{', else first '{')
    m = re.search(r'\n\s*\n\{', content)
    if m:
        start = m.start() + content[m.start():m.end()].rfind('{')
    else:
        start = content.find('{')

    block = None
    if start != -1:
        block, _ = _find_balanced_brace_block(content, start)

    # 3) try to parse the block (or fall back to direct extraction)
    parsed = None
    if block:
        try:
            parsed = ast.literal_eval(block)   # tolerant of single quotes
        except Exception:
            try:
                import json
                blk2 = block.replace("'", '"')
                parsed = json.loads(blk2)
            except Exception:
                parsed = None

    # 4) get value (prefer parsed dict; else direct regex in block; else direct regex in entire content)
    val_str = None
    if isinstance(parsed, dict) and key in parsed:
        val = parsed[key]
        val_str = str(val)
    else:
        # try direct extraction from block text first
        if block is not None:
            val_str = json_any_string_value(block, key)
        if not isinstance(val_str, str):
            # final fallback: whole content
            val_str = json_any_string_value(content, key) or ""

    # 5) if it's a string that encodes a list and allowed → parse & join
    fs = val_str.strip()
    if parse_list_like_if_string and fs.startswith("[") and fs.endswith("]"):
        try:
            cf_parsed = ast.literal_eval(fs)
            if isinstance(cf_parsed, (list, tuple, set)):
                val_str = " ".join(map(str, cf_parsed))
        except Exception:
            pass

    return normalize(val_str)

# =============== Rewards (no duplication) ===============f

def _cpy_reward_against_field(p_item, comp_text,
                              key: str,
                              tau=0.6, lam=0.0, gamma=2.0, scale_r=1,
                              parse_list_like_if_string=True):
    """
    Generic copy penalty reward against a named field in the prompt.
    Returns a dict with overlap, reward, and the extracted base text.
    """
    base_text = extract_field_string_from_prompt(
        p_item, key=key, parse_list_like_if_string=parse_list_like_if_string
    )
    if base_text == "":
        ov3 = 0.0
        reward = 1.0  # treat as no-copying when field missing
        scaled_reward = reward  # keep unscaled when field absent (as in your earlier code)
    else:
        ov3 = trigram_overlap(comp_text, base_text)
        reward = copy_signed_reward_trigram(ov3, tau=tau, lam=lam, gamma=gamma)
        scaled_reward = reward * scale_r

    return scaled_reward

def couns_cpy(p_item, comp_text, tau=0.6, lam=1.0, gamma=2.0, scale_r=1):
    """Copy penalty vs counselor_question."""
    return _cpy_reward_against_field(
        p_item, comp_text,
        key="counselor_question",
        tau=tau, lam=lam, gamma=gamma, scale_r=scale_r,
        parse_list_like_if_string=False,
    )

def feat_cpy(p_item, comp_text, tau=0.6, lam=1.0, gamma=2.0, scale_r=1):
    """Copy penalty vs cognitive_feature_value (allows parsing list-like strings)."""
    return _cpy_reward_against_field(
        p_item, comp_text,
        key="cognitive_feature_value",
        tau=tau, lam=lam, gamma=gamma, scale_r=scale_r,
        parse_list_like_if_string=True,
    )

rewards_featcpy = []
prompts_featcpy = []
completions_featcpy = []

rewards_counsCpy = []
prompts_counsCpy = []
completions_counsCpy = []


def featRwd(prompts, completions, **kwargs):
    rewards = []
    for p, c in zip(prompts, completions):
        res = feat_cpy(p, _as_text(c))  # returns float
        r = float(res["reward"]) if isinstance(res, dict) else float(res)
        rewards.append(r)
        rewards_featcpy.append(r)
        prompts_featcpy.append(p)
        completions_featcpy.append(c)
    print('featRwd===========:::::::::', rewards )
    return rewards


def cpyRwd(prompts, completions, **kwargs):
    rewards = []
    for p, c in zip(prompts, completions):
        res = couns_cpy(p, _as_text(c))  # returns float
        r = float(res["reward"]) if isinstance(res, dict) else float(res)
        rewards.append(r)
        rewards_counsCpy.append(r)
        prompts_counsCpy.append(p)
        completions_counsCpy.append(c)
    print('cpyRwd===========:::::::::', rewards )

    return rewards

#=============cpy end

rewards_len = []
prompts_len = []
completions_len = []
Length_len = []

def lenRwd(prompts, completions, **kwargs):


    low_cut = 30
    mid = 40
    high_sat = 130

    def extract_text(item):
        # string
        if isinstance(item, str):
            return item
        # single message dict
        if isinstance(item, dict):
            v = item.get("content")
            return v if isinstance(v, str) else ""
        # list of messages (your case)
        if isinstance(item, list) and item:
            # prefer an assistant message
            for m in item:
                if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                    return m["content"]
            # fallback: any message with content
            for m in item:
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    return m["content"]
            # nested fallback: stringify everything
            return " ".join(extract_text(x) for x in item)
        # last resort
        return str(item or "")

    rewards = []

    for p, comp in zip(prompts, completions):
        text = extract_text(comp)
        # normalize whitespace before counting words
        text = re.sub(r"\s+", " ", text).strip()
        L = len(text.split()) if text else 0

        if L <= low_cut:
            r = -1.0
        elif L < mid:
            # maps [low_cut, mid) -> [-1, 0)
            r = (L - mid) / float(mid - low_cut)   # e.g., (L-40)/10
        elif L <= high_sat:
            # maps [mid, high_sat] -> [0, 1]
            r = (L - mid) / float(high_sat - mid)  # e.g., (L-40)/90
        else:
            r = 1.0


        rewards.append(float(r*0.10))
        rewards_len.append(float(r*0.10))
        Length_len.append(L)
        completions_len.append(comp)
        prompts_len.append(p)
        print("length rewards L:::::::::=========>>>>", L)


    return rewards

#===========


# Dummy reward function for demonstration purposes
def reward_num_unique_letters(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]


#---------------------------------------------------------------new data end

def reward_by_length(prompts, completions, **kwargs):
    """
    Reward completions based on their character length.
    - Penalize if too short (< min_len chars).
    - Reward highest if close to target_len.
    - Scale reward smoothly between -1.0 and +1.0.

    Args:
        prompts (list[str]): List of prompts (unused here but kept for API compatibility).
        completions (list[str]): List of completion strings.
        **kwargs: Extra args (ignored).

    Returns:
        list[float]: Rewards for each completion.
    """
    rewards = []
    target_len = 128
    min_len = 4

    for p, c in zip(prompts, completions):
        length = 5

        if length == 0 or length < min_len:
            reward = -1.0   # bad: empty or too short
        else:
            # Reward peaks at target_len, decreases as you move away
            reward = 1.0 - abs(length - target_len) / target_len
            reward = max(-1.0, min(1.0, reward))  # clamp to [-1, 1]

        rewards.append(reward)

    return rewards

# ------------------ (Optional) Pretty live table of training metrics ------------------
class PandasStreamTableCallback(TrainerCallback):
    def __init__(self, refresh_every=1, save_csv="grpo_progress_table_new.csv"):
        self.rows = []; self.refresh_every = refresh_every; self.save_csv = save_csv
        self.columns = [
            ("Step",               lambda logs, step: step),
            ("Training Loss",     lambda logs, step: logs.get("loss", logs.get("train_loss"))),
            ("reward",            lambda logs, step: logs.get("reward", logs.get("reward/mean", logs.get("rewards/mean")))),
            ("reward_std",        lambda logs, step: logs.get("reward_std", logs.get("reward/std", logs.get("rewards/std")))),
            ("completion_length", lambda logs, step: logs.get("completions/mean_length")),
            ("kl",                lambda logs, step: logs.get("kl") or logs.get("kl_div") or logs.get("kl/mean")),
            ("rewards / attnReward", lambda logs, step: logs.get("rewards/attnReward/mean")),
            ("num_tokens",        lambda logs, step: logs.get("num_tokens")),
            ("lr",                lambda logs, step: logs.get("learning_rate")),
        ]
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        step = int(state.global_step)
        row = {}
        for name, fn in self.columns:
            try: row[name] = fn(logs, step)
            except Exception: row[name] = None
        self.rows.append(row)
        if len(self.rows) % self.refresh_every == 0:
            clear_output(wait=True)
            df = pd.DataFrame(self.rows, columns=[c[0] for c in self.columns])
            with pd.option_context("display.max_rows", 20, "display.width", 160, "display.precision", 6):
                display(df)
    def on_train_end(self, args, state, control, **kwargs):
        if recorder.buffer:
            recorder.flush()  # final ensure
        if self.save_csv and self.rows:
            pd.DataFrame(self.rows, columns=[c[0] for c in self.columns]).to_csv(self.save_csv, index=False)
            print(f"Saved log table to: {self.save_csv}")

# ------------------ Example GRPO config & trainer ------------------
training_args = GRPOConfig(
    output_dir="/N/slate/tnaduvil/models_save/Dial_grpo_new",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=625,
    max_completion_length=150,
    num_generations=4,
    optim="adamw_8bit",
    num_train_epochs=0.1,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    logging_steps=1,
    logging_strategy="steps",
    disable_tqdm=True,
    beta=0.1,   # weight for KL penaltybeta=0.1,   # weight for KL penalty
)


trainer = GRPOTrainer(
    model=model,
    reward_funcs=[attnReward, lenRwd],                      #,attnReward featRwd, lenRwd, cpyRwd <-- records prompts, completions, rewards
    args=training_args,
    train_dataset=dataset_train,
    callbacks=[PandasStreamTableCallback(refresh_every=1)],
)

trainer.train()  # uncomment to run


df = recorder.load_df()
display(df.head())

#############===================########save attention reward----
# Create DataFrame
df_grpo  = pd.DataFrame({
    "prompt": prompts_list,
    "completion": completions_list,
    "reward": rewards_list,
})

# Preview first few rows
print(df.head())

# Save to CSV if you like
df_grpo.to_csv("models_out/GRPO_training_generations_grpo_attention_new.csv", index=False)
print("training_generations_grpo:", len(df_grpo), "rows")


#############===================#length reward save--------

# Create DataFrame
df_grpo_len  = pd.DataFrame({
    "prompt": prompts_len,
    "completion": completions_len,
    "reward": rewards_len,
})

# Preview first few rows
print(df_grpo_len.head())

# Save to CSV if you like
df_grpo_len.to_csv("models_out/GRPO_training_generations_grpo_length_new.csv", index=False)
print("training_generations_grpo:", len(df_grpo_len), "rows")


#############===================#copy reward save--------


########feature copy
df_grpo_featCpy  = pd.DataFrame({
    "prompt": prompts_featcpy,
    "completion": completions_featcpy,
    "reward": rewards_featcpy,
})

# Preview first few rows
print(df_grpo_featCpy.head())

# Save to CSV if you like
print("training_generations_grpo:", len(df_grpo_featCpy), "rows")


########counselorQ copy

df_grpo_counCpy  = pd.DataFrame({
    "prompt": prompts_counsCpy,
    "completion": completions_counsCpy,
    "reward": rewards_counsCpy,
})

# Preview first few rows
print(df_grpo_counCpy.head())

# Save to CSV if you like
print("training_generations_grpo_len:", len(df_grpo_counCpy), "rows")


##########========================================================train stats--------------------------------- 
# Step 1: Extract logs
log_history = trainer.state.log_history  # list of dicts

# Step 2: Convert to DataFrame
df_logs = pd.DataFrame(log_history)
print('df_logs length', len(df_logs))
df_logs.to_csv("models_out/GRPO_trainer_stats_grpo_attn_len_90_new.csv", index=False)##########=====================DEBUG-update- train=========################

print("Training statss saved to trainer_stats_grpo_beta2.csv", len(df_logs), "rows")


model.save_pretrained("models_save/Dial_grpo/dial_grpo_lora_model_grpo")  # Local saving
tokenizer.save_pretrained("models_save/Dial_grpo/dial_grpo_lora_model_grpo")


##########========================================================test inference--------------------------------- 

#===========saving inference
import pandas as pd
import torch
from tqdm import tqdm
results = []
for i in tqdm(range(len(dataset_test))):
    # Build chat-formatted text
    print('inference saving....', i)
    text = tokenizer.apply_chat_template(
        dataset_test[i]['prompt'],
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and move to device
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=False
        )

    # Extract generated tokens
    full = gen_out.sequences[0]
    prompt_len = inputs["input_ids"].shape[1]
    gen_only_ids = full[prompt_len:]

    # Decode just assistant response
    assistant_text = tokenizer.decode(gen_only_ids, skip_special_tokens=True).lstrip()

    results.append({
        "prompt": dataset_test[i]['prompt'],
        "completion": assistant_text
    })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("models_out/GRPO_test_inference_results_grpo_attn_len_90_new.csv", index=False)  ###########=====================DEBUG-update test=========################

print("Saved", len(results), "rows to test_inference_results.csv")


#============


#================================Inference============================
