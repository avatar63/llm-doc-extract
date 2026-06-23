import anthropic
import json
import time
from typing import Optional, List
from pydantic import BaseModel, ValidationError, field_validator
import os
from dotenv import load_dotenv

# Looks for a .env file in the same directory
load_dotenv() 


# ── Schema ──────────────────────────────────────────────────────────────

class LineItem(BaseModel):
    item_name: str
    quantity: Optional[float] = None
    price: Optional[float] = None

class CanonicalRecord(BaseModel):
    company_name: Optional[str] = None
    address: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[float] = None
    line_items: List[LineItem] = []

    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v):
        if v is None:
            return v
        import re
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError(f"Date must be YYYY-MM-DD, got: {v}")
        return v


# ── Prompt ──────────────────────────────────────────────────────────────

NORMALIZATION_PROMPT = """Extract structured information from this receipt/invoice text and return as JSON.

Schema:
{{
  "company_name": string or null,
  "address": string or null,
  "date": string in YYYY-MM-DD format or null,
  "total_amount": float or null,
  "line_items": [{{"item_name": string, "quantity": float, "price": float}}]
}}

Rules:
- Use null for any field that cannot be confidently determined
- Dates must be YYYY-MM-DD format
- Amounts must be plain floats, no currency symbols
- Return ONLY the JSON object, no explanation, no markdown fences

Text:
{clean_text}"""


# ── Normalize single record ──────────────────────────────────────────────

def normalize_record(client, record, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": NORMALIZATION_PROMPT.format(
                        clean_text=record["clean_text"]
                    )
                }]
            )
            raw_output = response.content[0].text.strip()

            if raw_output.startswith("```"):
                raw_output = raw_output.strip("`")
                if raw_output.startswith("json"):
                    raw_output = raw_output[4:]
                raw_output = raw_output.strip()

            parsed = json.loads(raw_output)
            validated = CanonicalRecord(**parsed)

            return {
                "success": True,
                "source": record["source"],
                "x": record["noisy_text"],      # noisy input — model input at training
                "y": validated.model_dump(),     # clean canonical JSON — ground truth
                "intensity": record["intensity"]
            }

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return {
                "success": False,
                "source": record["source"],
                "error": str(e),
                "clean_text": record["clean_text"]
            }

        except anthropic.APIError as e:
            if attempt < max_retries:
                time.sleep(5)
                continue
            return {
                "success": False,
                "source": record["source"],
                "error": f"API error: {str(e)}",
                "clean_text": record["clean_text"]
            }

    return {"success": False, "error": "max retries exceeded"}


# ── Main batch run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    client = anthropic.Anthropic()

    records = []
    with open("combined_noisy.jsonl", "r") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Total records to process: {len(records)}")

    # Run validation batch first — inspect before full run
    VALIDATE_FIRST = False
    VALIDATION_SIZE = 25

    if VALIDATE_FIRST:
        records = records[:VALIDATION_SIZE]
        print(f"Running validation batch of {VALIDATION_SIZE} records first...")

    results = []
    failures = []

    for i, record in enumerate(records):
        result = normalize_record(client, record)

        if result["success"]:
            results.append(result)
        else:
            failures.append(result)

        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            print(f"Processed {i + 1}/{len(records)} | "
                  f"Success: {len(results)} | Failures: {len(failures)}")

        time.sleep(0.1)

    # Save results
    with open("xy_pairs.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    if failures:
        with open("xy_failures.jsonl", "w") as f:
            for r in failures:
                f.write(json.dumps(r) + "\n")

    success_rate = len(results) / len(records) * 100
    print(f"\nDone. Success: {len(results)} | Failures: {len(failures)} | "
          f"Rate: {success_rate:.1f}%")

    if VALIDATE_FIRST:
        print("\nValidation batch complete.")
        print("Inspect xy_pairs.jsonl manually, then set VALIDATE_FIRST = False and rerun.")