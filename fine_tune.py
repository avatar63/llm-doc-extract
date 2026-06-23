"""
LoRA Fine-tuning Script for Document Structured Extraction
Base model: Qwen2.5-0.5B-Instruct
Task: Noisy OCR text -> Canonical JSON extraction
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ── Config ───────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./doc_extract_lora"
TRAINING_DATA = "./dataset/training_data.jsonl"

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
}

TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,  # effective batch size = 16
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "fp16": True,
    "logging_steps": 50,
    "eval_strategy": "steps",
    "eval_steps": 200,
    "save_strategy": "steps",
    "save_steps": 200,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "max_seq_length": 1024
}


# ── Load and format dataset ──────────────────────────────────────────────

def load_dataset_from_jsonl(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def format_for_qwen(record, tokenizer):
    """
    Format instruction/input/output into Qwen2.5's chat template.
    Loss is computed only on the output (assistant) tokens.
    """
    messages = [
        {
            "role": "system",
            "content": record["instruction"]
        },
        {
            "role": "user",
            "content": record["input"]
        },
        {
            "role": "assistant",
            "content": record["output"]
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print(f"Loading tokenizer and model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # ── Apply LoRA ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load and format data ─────────────────────────────────────────────
    print("Loading training data...")
    raw_records = load_dataset_from_jsonl(TRAINING_DATA)

    # Format using Qwen's chat template
    formatted_texts = [
        format_for_qwen(r, tokenizer) for r in raw_records
    ]

    # Train/val split — 90/10
    split_idx = int(len(formatted_texts) * 0.9)
    train_texts = formatted_texts[:split_idx]
    val_texts = formatted_texts[split_idx:]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── Training arguments ───────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        fp16=TRAINING_CONFIG["fp16"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        eval_strategy=TRAINING_CONFIG["eval_strategy"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        load_best_model_at_end=TRAINING_CONFIG["load_best_model_at_end"],
        dataset_text_field="text",
        report_to="none",
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer

    )

    # ── Train ────────────────────────────────────────────────────────────
    print("Starting training...")
    trainer.train()

    # ── Save adapter ─────────────────────────────────────────────────────
    print(f"Saving LoRA adapter to {OUTPUT_DIR}/final_adapter")
    model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

    print("Training complete.")


if __name__ == "__main__":
    main()