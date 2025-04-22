#!/usr/bin/env python
"""
train_distilbert.py
-------------------
Finetune DistilBERT on tokenized sarcasm dataset.
"""

import argparse
from pathlib import Path
import numpy as np

from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    """Return accuracy and macro‑F1 for Hugging Face Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    # -------- 1. CLI arguments --------
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to tokenized dataset")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="ckpt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    # -------- 2. Load dataset --------
    ds = load_from_disk(args.data_dir)

    # -------- 3. Load tokenizer & model --------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # -------- 4. TrainingArguments --------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        logging_dir="tb_logs",          
        save_strategy="epoch",
        report_to="tensorboard",
        load_best_model_at_end=True,
        seed=42,
    )

    # -------- 5. Data collator --------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------- 6. Create Trainer --------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # -------- 7. Final evaluation on test --------
    metrics = trainer.evaluate(ds["test"])
    print("✅ Finetuning complete! Test metrics:", metrics)

    # Save everything
    trainer.save_model(Path(args.output_dir) / "best_model")
    (Path(args.output_dir) / "metrics.json").write_text(str(metrics))


if __name__ == "__main__":
    main()