"""
Evaluation script — baseline vs fine-tuned model
Field-level accuracy on held-out val set
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from collections import defaultdict
import re

# ── Config ───────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./doc_extract_lora/final_adapter"
VAL_DATA = "./dataset/xy_pairs.jsonl"
EVAL_SAMPLES = 204  # start with 50, expand to 204 after confirming it works

INSTRUCTION = (
    "Extract the following fields from the OCR text as JSON: "
    "company_name, address, date, total_amount, line_items "
    "(each with item_name, quantity, price). "
    "Use null for any field that cannot be determined."
)


# ── Load val data ────────────────────────────────────────────────────────

def load_val_data(filepath, n_samples=None):
    records = []
    with open(filepath, "r") as f:
        for line in f:
            records.append(json.loads(line))

    # Use last 10% as val (matches training split)
    split_idx = int(len(records) * 0.9)
    val_records = records[split_idx:]

    if n_samples:
        val_records = val_records[:n_samples]

    print(f"Evaluating on {len(val_records)} samples")
    return val_records


# ── Model loading ────────────────────────────────────────────────────────

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH, trust_remote_code=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


# ── Inference ────────────────────────────────────────────────────────────

def extract(model, tokenizer, noisy_text):
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": noisy_text}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        # Handle case where model returns a list instead of a dict
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else None
        return parsed
    except json.JSONDecodeError:
        return None


# ── Field-level evaluation ───────────────────────────────────────────────

def normalize_string(s):
    """Lowercase, strip, remove extra whitespace for fuzzy string matching."""
    if s is None:
        return None
    return re.sub(r'\s+', ' ', str(s).lower().strip())

def eval_company_name(pred, gt):
    """Fuzzy match — correct if predicted contains ground truth or vice versa."""
    if pred is None and gt is None:
        return True
    if pred is None or gt is None:
        return False
    p = normalize_string(pred)
    g = normalize_string(gt)
    return g in p or p in g

def eval_date(pred, gt):
    """Exact match on YYYY-MM-DD."""
    if pred is None and gt is None:
        return True
    if pred is None or gt is None:
        return False
    return str(pred).strip() == str(gt).strip()

def eval_total_amount(pred, gt):
    """Numeric match within 1% tolerance."""
    if pred is None and gt is None:
        return True
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) / (float(gt) + 1e-9) < 0.01
    except (ValueError, TypeError):
        return False

def eval_line_items(pred, gt):
    """Returns fraction of ground truth items correctly identified."""
    if not gt:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    return min(len(pred), len(gt)) / len(gt)

def eval_json_valid(pred):
    """Did the model return parseable JSON at all."""
    return pred is not None

def evaluate_record(pred, gt):
    if isinstance(pred, list):
        pred = pred[0] if pred else None
    return {
        "json_valid": eval_json_valid(pred),
        "company_name": eval_company_name(
            pred.get("company_name") if pred else None,
            gt.get("company_name")
        ),
        "date": eval_date(
            pred.get("date") if pred else None,
            gt.get("date")
        ),
        "total_amount": eval_total_amount(
            pred.get("total_amount") if pred else None,
            gt.get("total_amount")
        ),
        "line_items": eval_line_items(
            pred.get("line_items") if pred else [],
            gt.get("line_items", [])
        ),
    }


# ── Run evaluation ───────────────────────────────────────────────────────

def run_eval(model, tokenizer, val_records, label):
    print(f"\nRunning evaluation: {label}")
    scores = defaultdict(list)

    for i, record in enumerate(val_records):
        noisy_text = record["x"]
        ground_truth = record["y"]

        pred = extract(model, tokenizer, noisy_text)
        result = evaluate_record(pred, ground_truth)

        for field, score in result.items():
            scores[field].append(float(score))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(val_records)} done...")

    print(f"\n{'='*50}")
    print(f"RESULTS — {label}")
    print(f"{'='*50}")
    for field, field_scores in scores.items():
        avg = sum(field_scores) / len(field_scores) * 100
        print(f"  {field:<20} {avg:.1f}%")
    print(f"{'='*50}\n")

    return scores


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    val_records = load_val_data(VAL_DATA, n_samples=EVAL_SAMPLES)

    # Evaluate baseline
    base_model, base_tokenizer = load_base_model()
    base_scores = run_eval(base_model, base_tokenizer, val_records, "BASELINE")
    del base_model
    torch.cuda.empty_cache()

    # Evaluate fine-tuned
    ft_model, ft_tokenizer = load_finetuned_model()
    ft_scores = run_eval(ft_model, ft_tokenizer, val_records, "FINE-TUNED")
    del ft_model
    torch.cuda.empty_cache()

    # Side by side comparison
    print(f"\n{'='*50}")
    print("COMPARISON — BASELINE vs FINE-TUNED")
    print(f"{'='*50}")
    print(f"{'Field':<20} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>8}")
    print(f"{'-'*50}")
    for field in base_scores:
        base_avg = sum(base_scores[field]) / len(base_scores[field]) * 100
        ft_avg = sum(ft_scores[field]) / len(ft_scores[field]) * 100
        delta = ft_avg - base_avg
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"  {field:<18} {base_avg:>9.1f}% {ft_avg:>11.1f}% {arrow}{abs(delta):>6.1f}%")
    print(f"{'='*50}")