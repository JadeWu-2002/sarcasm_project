#!/usr/bin/env python
"""
zero_shot.py
------------
Run zero‑shot sarcasm detection with flan‑t5-small.
"""

import argparse
import json
import numpy as np

from pathlib import Path
from datasets import load_from_disk
from evaluate import load as load_metric # type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main() -> None:
    # ---------- 1. CLI ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to *raw* DatasetDict")
    parser.add_argument("--out_file", default="results/zero_shot.json")
    args = parser.parse_args()

    # ---------- 2. Load test set ----------
    ds = load_from_disk(args.data_dir)["test"]   # expects columns: text, label

    # ---------- 3. Load flan‑T5 ----------
    model_name = "google/flan-t5-small"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # ---------- 4. Prepare metrics ----------
    accuracy = load_metric("accuracy")
    f1       = load_metric("f1")

    # ---------- 5. Zero‑shot inference ----------
    preds = []
    for ex in ds:
        prompt = f'Text: "{ex["text"]}"\n\nIs this headline sarcastic? Answer Yes or No.'
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=2)
        answer = tokenizer.decode(output[0], skip_special_tokens=True).lower()
        pred   = 1 if "yes" in answer else 0
        preds.append(pred)

        accuracy.add(prediction=pred, reference=ex["label"])
        f1.add(prediction=pred, reference=ex["label"])

    # ---------- 6. Save & print ----------
    res = {
        "accuracy": accuracy.compute()["accuracy"],
        "f1":       f1.compute(average="macro")["f1"],
    }
    print("✅ Zero‑shot metrics:", res)

    Path(args.out_file).parent.mkdir(exist_ok=True)
    Path(args.out_file).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()