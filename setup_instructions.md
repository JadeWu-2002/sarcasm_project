# 1. Create and activate a Python virtual environment
```
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
# 2. Upgrade pip to the latest version
```
pip install --upgrade pip
```
# 3. Install required dependencies
```
pip install transformers>=4.40.0
pip install datasets>=2.18.0
pip install evaluate>=0.4.1
pip install torch>=2.0
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install tensorboard
```

# 4. (Optional) Install additional package if prompted during zero-shot execution
```
pip install huggingface_hub[hf_xet]
```
# 5. Launch TensorBoard to visualize training logs
```
tensorboard --logdir=tb_logs
```

# 6. Example commands to run the project

# Data preprocessing
```
python prepare_data.py --input data/Sarcasm_Headlines_Dataset.json --output_dir data
```
# Fine-tune DistilBERT
```
python train_distilbert.py --data_dir data/tokenized --output_dir ckpt --epochs 2 --lr 2e-5
```
# Perform Zero-Shot inference
```
python zero_shot.py --data_dir data/raw --out_file results/zero_shot.json
```
# Evaluate the fine-tuned model
```
python evaluate.py --data_dir data/tokenized --ckpt ckpt/best_model --out_file results/finetune.json
```