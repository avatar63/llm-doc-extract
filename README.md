# llm-doc-extract

LoRA fine-tuned Qwen2.5-0.5B for structured JSON extraction from noisy OCR receipts and invoices.

---

## What it does

Takes raw, noisy OCR text from scanned receipts and invoices and extracts structured JSON — company name, address, date, total amount, and line items — reliably, locally, with no API calls at inference time.

```
Input (noisy OCR):                          Output (structured JSON):
RELI4NCE FR3SH                    →         {
Sh0p N0 12, 5ect0r 18                         "company_name": "RELIANCE FRESH SHOP",
D4te: O5-ll-2O24                              "address": "NO 12, SECTOR 18, GURUGRAM",
Net P4y4ble: 34O.OO                           "date": "2024-11-05",
                                              "total_amount": 340.0,
                                              "line_items": [...]
                                            }
```

---

## Why local inference

Most document extraction pipelines make repeated API calls — every document processed costs money and sends your data to a third-party server. This project uses LLM-assisted labeling as a **one-time dataset generation cost**, then fine-tunes a small local model to internalize that extraction capability.

At inference time: zero API cost, zero network latency, zero data leaving your machine.

---

## Results

Evaluated on 204 held-out examples, comparing base Qwen2.5-0.5B-Instruct against the fine-tuned version:

| Field | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| JSON valid | 80.9% | 99.5% | +18.6% |
| Company name | 38.2% | 46.6% | +8.3% |
| Date | 15.2% | 83.8% | +68.6% |
| Total amount | 0.0% | 99.0% | +99.0% |
| Line items | 58.5% | 97.0% | +38.5% |

The baseline frequently produced malformed, unparseable JSON and hallucinated values. The fine-tuned model produces valid, schema-conformant JSON 99.5% of the time.

---

## Architecture

```
Scanned image
     ↓
  docTR OCR
     ↓
 Noisy text  ──────────────────────────────────────┐
     ↓                                              │
 [Training only]                                   │ [Inference]
 Noise injection                                   │
     ↓                                             ↓
 Claude Haiku   →   Canonical JSON (Y)    Fine-tuned Qwen2.5-0.5B
     ↓                                             ↓
 (X, Y) pairs                              Structured JSON output
     ↓
 LoRA fine-tuning
```

---

## Dataset pipeline

Two public datasets combined:
- **SROIE** (ICDAR 2019) — 626 real scanned receipts, bounding-box format
- **Voxel51 / Kaggle** — 1,414 synthetic invoice images with OCR text

**Pipeline steps:**
1. `data/prepare_sroie.py` — extract text from SROIE bounding-box format
2. `data/prepare_voxel.py` — extract OCR text from Voxel51 dataset
3. `data/merge.py` — combine into single dataset
4. `data/create_noisy_data.py` — synthetic noise injection at light/medium/heavy intensity
5. `data/normalize_labels.py` — Claude Haiku generates canonical JSON ground truth from clean text (~$0.50 total for ~2040 records)
6. `data/format_training_data.py` — format as instruction-tuning pairs

Total: ~2040 (noisy text, canonical JSON) training pairs.

---

## Training

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Method | LoRA (PEFT + TRL SFTTrainer) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 8.8M / 502M (1.75%) |
| Epochs | 3 |
| Batch size | 4 (effective 16 with gradient accumulation) |
| Learning rate | 2e-4 (cosine schedule) |
| Hardware | RTX 3060 12GB |
| Training time | ~28 minutes |
| Final train loss | 1.45 |
| Final eval loss | 1.31 |

---

## Usage

**Install dependencies:**
```bash
pip install transformers peft torch
```

**Run inference:**
```python
from inference import load_finetuned_model, extract

model, tokenizer = load_finetuned_model()

noisy_ocr_text = """
RELI4NCE FR3SH
Sh0p N0 12, 5ect0r 18
D4te: O5-ll-2O24
Net P4y4ble: 34O.OO
"""

result = extract(model, tokenizer, noisy_ocr_text)
print(result)
```

**Model adapter on HuggingFace:** [avatar63/qwen-receipt-extractor](https://huggingface.co/avatar63/qwen-receipt-extractor)

---

## Limitations

- **Character-level denoising is partial** — the model handles most OCR noise but some character substitutions in item names and company suffixes remain. A lightweight rule-based normalization pre-processing step addresses this.
- **Address hallucination** — when address information is sparse or ambiguous, the model occasionally generates plausible-looking but incorrect addresses rather than returning null. Future work includes adding explicit null-address training examples.
- **Net payable vs subtotal ambiguity** — on some receipts the model extracts the pre-discount subtotal rather than the final payable amount.
- **Synthetic training data** — the Voxel51 dataset is synthetically generated. Real-world diversity may require augmentation with additional real scanned documents.

---

## Future work

- Rule-based character normalization pre-processing layer
- ONNX export and quantization for edge/mobile deployment
- Expand to Hindi and Arabic receipts (multilingual extension)
- Increase training data volume for improved company name denoising

---

## Acknowledgements

**Datasets:**
- [SROIE Dataset](https://rrc.cvc.uab.es/?ch=13) — ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction
- [High Quality Invoice Images for OCR](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr) — Osama Hosam Abdellatif via Kaggle, FiftyOne port by Harpreet Sahota

**Base model:**
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — Qwen Team, Alibaba Cloud

**Labeling:**
- Ground truth labels generated using [Claude Haiku](https://www.anthropic.com/claude) (Anthropic) via LLM-assisted data labeling

---

## Stack
`PyTorch` · `HuggingFace Transformers` · `PEFT` · `TRL` · `Anthropic Claude Haiku` · `docTR` · `Pydantic`



