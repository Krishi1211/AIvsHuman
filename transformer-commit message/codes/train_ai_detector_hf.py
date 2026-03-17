import json
import glob
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Probabilities for class 1 (AI)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    ai_probs = probs[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    # MSE as Brier Score (for binary, use predicted probability vs actual label)
    mse = mean_squared_error(labels, ai_probs)
    
    # R2 Score mapping explained variance
    r2 = r2_score(labels, ai_probs)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mse': mse,
        'r2': r2
    }

def load_data():
    base_path = '/Users/krishi1211/Documents/SE/AIvsHuman/detect-gpt-fork/results'
    files = glob.glob(f'{base_path}/*/*/raw_data.json', recursive=True)
    
    ai_texts = []
    human_texts = []
    
    # Load AI Texts
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if 'sampled' in data:
                    for text in data['sampled']:
                        if isinstance(text, str) and len(text.strip()) > 5:
                            ai_texts.append(text)
            except Exception as e:
                pass
                
    # Load Human Texts specifically from Pydriller parquet additions
    parquet_path = '/Users/krishi1211/Documents/SE/AIvsHuman/Pydriller/commit_metrics.parquet'
    try:
        df_human = pd.read_parquet(parquet_path)
        human_texts = df_human['message'].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Could not load parquet: {e}")
        
    # Balance dataset
    ai_count = len(ai_texts)
    print(f"Loaded {ai_count} AI items from JSONs.")
    print(f"Loaded {len(human_texts)} Human items from Parquet.")
    
    # Ensure balance - sample human texts to match ai_texts
    if len(human_texts) > ai_count:
        import random
        random.seed(42)
        human_texts = random.sample(human_texts, ai_count)
        
    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)

    # Clean extreme outliers (long texts > 1000 characters to prevent memory blowout)
    texts_clean, labels_clean = [], []
    for t, l in zip(texts, labels):
        t_str = str(t)[:512]
        texts_clean.append(t_str)
        labels_clean.append(l)

    return texts_clean, labels_clean

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model from scratch')
    parser.add_argument('--predict', type=str, help='String inference test')
    args = parser.parse_args()

    model_dir = '/Users/krishi1211/Documents/SE/AIvsHuman/transformer-commit message/saved_hf_detector'

    if args.predict:
        print(f"Pydriller ML Inference -> {args.predict}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        inputs = tokenizer(args.predict, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            ai_prob = probs[0][1].item()
            label = "AI (1)" if ai_prob > 0.5 else "Human (0)"
            print(f"\nPrediction: {label} (AI Probability: {ai_prob:.4f})")
        return

    # FULL TRAINING LOOP
    print("Extracting Pydriller/JSON datasets...")
    texts, labels = load_data()
    print(f"Total Combined Array: {len(texts)} samples")
    
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_data = {'text': X_train, 'label': y_train}
    val_data = {'text': X_val, 'label': y_val}

    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized_train = train_dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=128), batched=True)
    tokenized_val = val_dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=128), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    training_args = TrainingArguments(
        output_dir='/Users/krishi1211/Documents/SE/AIvsHuman/transformer-commit message/results_hf',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_dir='/Users/krishi1211/Documents/SE/AIvsHuman/transformer-commit message/logs',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    print("Training HuggingFace Transformer model...")
    trainer.train()

    print("\nEvaluating against Unseen Pydriller Data...")
    metrics = trainer.evaluate()
    print(f"Eval metrics: {metrics}")

    print(f"Saving newly trained model to {model_dir}...")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == '__main__':
    main()
