import os
import json
import glob
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score

def load_data():
    base_path = '/Users/krishi1211/Documents/SE/AIvsHuman/detect-gpt-fork/results'
    files = glob.glob(f'{base_path}/*/*/raw_data.json', recursive=True)
    
    texts = []
    labels = []
    
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # 'original' = Human = 0
                if 'original' in data:
                    for text in data['original']:
                        texts.append(text)
                        labels.append(0)  # Human
                        
                # 'sampled' = AI = 1
                if 'sampled' in data:
                    for text in data['sampled']:
                        texts.append(text)
                        labels.append(1)  # AI
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                
    return texts, labels

def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(-1)
    
    # Calculate probabilities for class 1 to compute MSE and R^2
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    prob_class_1 = probs[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    mse = mean_squared_error(labels, prob_class_1)
    r2 = r2_score(labels, prob_class_1)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mse': mse,
        'r2': r2
    }

def train_hf_model():
    # 1. Load Data
    texts, labels = load_data()
    print(f"Loaded {len(texts)} samples")
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    # 2. Convert to HuggingFace Dataset
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    
    # 3. Load Tokenizer & Model
    model_name = "distilbert-base-uncased" # Faster training than roberta 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 4. Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir='./results_hf',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )
    
    print("Training HuggingFace model...")
    trainer.train()
    
    # 7. Evaluate and Save
    print("Evaluating...")
    results = trainer.evaluate()
    print("Eval metrics:", results)
    
    print("Saving model and tokenizer...")
    trainer.save_model("./saved_hf_detector")
    tokenizer.save_pretrained("./saved_hf_detector")

def predict_message(text, model_path="./saved_hf_detector"):
    # Load model for inference
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        
    label = "AI (1)" if pred_label == 1 else "Human (0)"
    
    # Standardize the returning score formatting between 0.0 and 1.0 (treating 1.0 as completely confident it's AI)
    ai_score = probs[0][1].item()
    
    # We strip down the printed text to just deliver the raw AI score and basic print format.
    print(f"\nMessage: '{text}' -> Prediction: {label}")
    print(f"AI Score: {ai_score:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_hf_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "--predict":
        msg = sys.argv[2] if len(sys.argv) > 2 else "Fix timezone display for logs on UI (#23075)"
        predict_message(msg)
    else:
        print("Usage:")
        print("  python train_ai_detector_hf.py --train")
        print("  python train_ai_detector_hf.py --predict \"My test commit message\"")
