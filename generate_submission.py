"""
generate_submission.py — builds submission.jsonl (one line per official test pair).

Reads the expanded dataset produced by:
    cd dataset && python generate_dataset.py --seed-dir . --out ./expanded
and calls bot.compose() for each of the 30 pairs in dataset/expanded/test_pairs.json.

Usage:
    python generate_submission.py                     # writes submission.jsonl
    python generate_submission.py --now 2026-04-26T10:30:00Z --out submission.jsonl

Design notes:
- `now` defaults to 2026-04-26T10:30:00Z, the reference date of the challenge
  brief: the trigger payloads' own day counts (e.g. days_to_wedding: 196) were
  computed on that date, so our computed counts agree with them.
- Offline there is no 15s budget, so when Groq's free-tier limit is hit we WAIT
  for the cooldown and try again (up to a few times) instead of settling for the
  template. If the LLM still isn't available, the validated template is used.
- Every line is checked for the required fields before the file is written.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import bot  # loads .env and exposes compose()
import llm

ROOT = Path(__file__).parent
REQUIRED = ("test_id", "body", "cta", "send_as", "suppression_key", "rationale")


def load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compose_with_patience(category, merchant, trigger, customer, now, attempts=4):
    """compose(), but if the LLM was only rate-limited, wait for the cooldown and retry."""
    out = bot.compose(category, merchant, trigger, customer, now)
    for _ in range(attempts):
        if "LLM-written" in out["rationale"] or not llm.enabled():
            break
        if "rate_limited" not in out["rationale"] and "http_429" not in out["rationale"]:
            break  # rejected by the validator or another error: keep the template
        wait = max(1.0, max(llm._cooldown_until.values(), default=0) - time.monotonic()) + 1
        print(f"    rate-limited, waiting {wait:.0f}s ...")
        time.sleep(min(wait, 60))
        out = bot.compose(category, merchant, trigger, customer, now)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset" / "expanded"))
    ap.add_argument("--now", default="2026-04-26T10:30:00Z")
    ap.add_argument("--out", default=str(ROOT / "submission.jsonl"))
    args = ap.parse_args()

    data = Path(args.data)
    if not (data / "test_pairs.json").exists():
        sys.exit("dataset/expanded not found - run: cd dataset && python generate_dataset.py --seed-dir . --out ./expanded")
    categories = {p.stem: load(p) for p in (data / "categories").glob("*.json")}
    pairs = load(data / "test_pairs.json")["pairs"]

    lines, sources = [], {"LLM": 0, "template": 0}
    for p in pairs:
        trigger = load(data / "triggers" / f"{p['trigger_id']}.json")
        merchant = load(data / "merchants" / f"{p['merchant_id']}.json")
        customer = load(data / "customers" / f"{p['customer_id']}.json") if p.get("customer_id") else None
        category = categories[merchant["category_slug"]]

        out = compose_with_patience(category, merchant, trigger, customer, args.now)
        src = "LLM" if "LLM-written" in out["rationale"] else "template"
        sources[src] += 1
        line = {"test_id": p["test_id"], "body": out["body"], "cta": out["cta"], "send_as": out["send_as"],
                "suppression_key": out["suppression_key"], "rationale": out["rationale"]}
        missing = [k for k in REQUIRED if not str(line.get(k) or "").strip()]
        if missing:
            sys.exit(f"{p['test_id']}: missing {missing}")
        lines.append(line)
        print(f"{p['test_id']} {trigger['kind']:<26} [{src}] {out['body'][:70]}...")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(lines)} lines to {args.out}  (LLM: {sources['LLM']}, template: {sources['template']})")


if __name__ == "__main__":
    main()
