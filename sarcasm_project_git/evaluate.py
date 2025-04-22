#!/usr/bin/env python
"""
evaluate.py
-----------
Load a fine‑tuned checkpoint, run prediction on the test set,
write metrics to JSON, and save a confusion‑matrix PNG.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def main() -> None:
    # ---------- 1. CLI ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="tokenized DatasetDict path")
    parser.add_argument("--ckpt", required=True, help="path to best_model directory")
    parser.add_argument("--out_file", default="results/finetune.json")
    parser.add_argument("--fig_dir", default="fig")
    args = parser.parse_args()

    # ---------- 2. Load test set ----------
    ds = load_from_disk(args.data_dir)["test"]

    # ---------- 3. Load model ----------
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model     = AutoModelForSequenceClassification.from_pretrained(args.ckpt)

    # ---------- 4. Batch predict ----------
    preds, labels = [], []
    for ex in ds:
        import torch
        inputs = {
            "input_ids": torch.tensor(ex["input_ids"])[None, :],
            "attention_mask": torch.tensor(ex["attention_mask"])[None, :]
        }
        logits = model(**inputs).logits
        preds.append(int(np.argmax(logits.detach().numpy(), axis=-1)))
        labels.append(int(ex["label"]))

    # ---------- 5. Compute metrics ----------
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro")
    res = {"accuracy": acc, "f1": f1}
    print("✅ Finetune metrics:", res)

    # ---------- 6. Confusion matrix ----------
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non‑sarcastic", "Sarcastic"])
    disp.plot(cmap="Blues", xticks_rotation=45)
    Path(args.fig_dir).mkdir(exist_ok=True)
    fig_path = Path(args.fig_dir) / "cm.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print("🖼  Confusion‑matrix saved to:", fig_path)

    # ---------- 7. Save metrics ----------
    Path(args.out_file).parent.mkdir(exist_ok=True)
    Path(args.out_file).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()