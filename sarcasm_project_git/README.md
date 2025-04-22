# Sarcasm Detection with DistilBERT
This project fine‑tunes Hugging Face’s DistilBERT to detect sarcasm in news headlines and benchmarks its performance against a zero‑shot flan‑T5‑small baseline.

## Main Files & Folders

```
sarcasm_project/
├── data/
│   ├── raw/                # train/val/test splits of the original headlines
│   └── tokenized/          # tokenized DatasetDict ready for training
├── ckpt/
│   └── best_model/         # best DistilBERT checkpoint after fine‑tuning
├── results/
│   ├── zero_shot.json      # zero‑shot flan‑T5‑small accuracy & F1
│   └── finetune.json       # fine‑tuned DistilBERT accuracy & F1
├── fig/
│   ├── cm.png              # confusion matrix for test‑set evaluation
│   └── tensorboard_eval_loss.png # validation loss curve from TensorBoard
├── prepare_data.py         # loads, splits, and tokenizes raw data
├── train_distilbert.py     # fine‑tunes DistilBERT on tokenized data
├── zero_shot.py            # runs zero‑shot inference with flan‑T5‑small
├── evaluate.py             # evaluates model & generates cm.png
├── requirements.txt        # pip‐installable list of dependencies
└── README.md               # project documentation
```


