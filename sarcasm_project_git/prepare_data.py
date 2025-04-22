#!/usr/bin/env python
"""
prepare_data.py
---------------
This script:
1. Reads a sarcasm‐headline dataset (CSV / JSON / JSONL)
2. Keeps only the text column (“headline”) and the label column (“is_sarcastic”)
3. Splits the data into train / validation / test
4. Tokenizes the text with a Hugging Face tokenizer
5. Saves both raw and tokenized DatasetDict objects to disk
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def read_data(path: str) -> pd.DataFrame:
    """Read a file into a pandas DataFrame depending on its extension."""
    ext = Path(path).suffix.lower()
    if ext in {".csv", ".tsv"}:
        return pd.read_csv(path)
    elif ext in {".json", ".jsonl"}:
        return pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def main() -> None:
    # ---------- 1. Parse command‑line arguments ----------
    parser = argparse.ArgumentParser(description="Pre‑process sarcasm headline data")
    parser.add_argument("--input", required=True, help="Path to raw data file")
    parser.add_argument("--output_dir", default="data", help="Where to save processed data")
    parser.add_argument("--model_name", default="distilbert-base-uncased",
                        help="Tokenizer model to use")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # ---------- 2. Read raw data ----------
    df = read_data(args.input)

    # Keep only two columns and rename them to match HF convention
    df = df[["headline", "is_sarcastic"]].rename(
        columns={"headline": "text", "is_sarcastic": "label"}
    )
    df["label"] = df["label"].astype(int)

    # ---------- 3. Convert to HF Dataset and split ----------
    raw_ds = Dataset.from_pandas(df, preserve_index=False).shuffle(seed=args.seed)

    tmp_split = raw_ds.train_test_split(test_size=args.test_size, seed=args.seed)
    # Further split test set into val + test according to val_size
    val_fraction = args.val_size / args.test_size
    val_test = tmp_split["test"].train_test_split(test_size=val_fraction, seed=args.seed)

    ds = DatasetDict({
        "train": tmp_split["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

    # ---------- 4. Tokenize texts ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,   # headlines are short; 128 tokens is enough
        )

    tokenized_ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # ---------- 5. Save datasets to disk ----------
    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = Path(args.output_dir) / "raw"
    tok_path = Path(args.output_dir) / "tokenized"

    ds.save_to_disk(raw_path)
    tokenized_ds.save_to_disk(tok_path)

    print("✅ Done! Saved raw data to:", raw_path)
    print("✅ Done! Saved tokenized data to:", tok_path)


if __name__ == "__main__":
    main()