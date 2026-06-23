"""
Quick inference test — compare baseline vs fine-tuned model
on noisy OCR text
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Config ───────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./doc_extract_lora/final_adapter"

INSTRUCTION = (
    "Extract the following fields from the OCR text as JSON: "
    "company_name, address, date, total_amount, line_items "
    "(each with item_name, quantity, price). "
    "Use null for any field that cannot be determined."
)

TEST_INPUTS = [
    """10 GRAM GOURMET SBN BHD\n(1152264-K)\n\nNO 3, JALAN TEMENG6UNG 27/9\nB_ANDAR MAHKOT4 CHERAS,\n4320OSEL4NGOR.\n(GST -REG. NO : 00Z055098368)\nTAX INV.OIC\nT_ABLE -\nCHECK #: 51-46S8\n\nPAX(S): O\nDATE\n: 11-O6-2018 12:51:34\n\nCASHIER:.    CASHIE M0RNING\nDESCIPTION\nQTY\nUPRICE\nTOTAL TAX\n\nP02 SPAGHETTI 4GLIO OLIO CICKEN BRE4S-T\n1 X\n1S.00\n15.0O\n\nSR\nTOTAL (EXCLUDING GST):\n\n15.O0\nTOT4L GST (0%):\n0.00\nTOT4L (INCLUSIVE OF GST):\nI5.00\nTOTAL:\n15.O,0\n\nCLOSED: 8888\n11-06-20I8\nl2:52:21\nSERVE: CA5,HIER MORNING\nCASH :`\n\n15.00\nG5T    SUMMARY\nA_MOUNT(RM)\nTAX(RM)\nSR\n\n(@ 0%)\n15.00\n\n0.0O\nTH'ANK YOU\nPLE4SE COME AGAIN"""
]


# ── Load model ───────────────────────────────────────────────────────────
def load_base_model():
    print("Loading base model (no adapter)...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer



def load_finetuned_model():
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH,
        trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


# ── Inference ────────────────────────────────────────────────────────────

def extract(model, tokenizer, noisy_text):
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": noisy_text}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,   # low temperature for deterministic extraction
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True)

    # Try to parse as JSON
    try:
        parsed = json.loads(raw_output.strip())
        return parsed, raw_output
    except json.JSONDecodeError:
        return None, raw_output


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    for i, test_input in enumerate(TEST_INPUTS):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}")
        print(f"{'='*60}")
        print(f"INPUT (first 100 chars): {test_input[:100]}...\n")

        # Base model
        print("--- BASELINE (no adapter) ---")
        base_model, base_tokenizer = load_base_model()
        parsed_base, raw_base = extract(base_model, base_tokenizer, test_input)
        print(f"Raw: {raw_base[:300]}")
        if parsed_base:
            print(f"Parsed: {json.dumps(parsed_base, indent=2)}")
        else:
            print("Failed to parse as JSON")

        # Free VRAM before loading fine-tuned
        del base_model
        torch.cuda.empty_cache()

        # Fine-tuned model
        print("\n--- FINE-TUNED ---")
        ft_model, ft_tokenizer = load_finetuned_model()
        parsed_ft, raw_ft = extract(ft_model, ft_tokenizer, test_input)
        print(f"Raw: {raw_ft[:300]}")
        if parsed_ft:
            print(f"Parsed: {json.dumps(parsed_ft, indent=2)}")
        else:
            print("Failed to parse as JSON")

        del ft_model
        torch.cuda.empty_cache()