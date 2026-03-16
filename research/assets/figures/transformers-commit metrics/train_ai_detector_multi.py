import json
import glob
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
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
                if 'original' in data:
                    for text in data['original']:
                        texts.append(text)
                        labels.append(0)  # Human
                        
                if 'sampled' in data:
                    for text in data['sampled']:
                        texts.append(text)
                        labels.append(1)  # AI
            except Exception as e:
                pass
                
    return texts, labels

def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    # Predict probabilities for MSE and R2
    probs = pipeline.predict_proba(X_test)[:, 1]
        
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    acc = accuracy_score(y_test, preds)
    mse = mean_squared_error(y_test, probs)
    r2 = r2_score(y_test, probs)
    
    return {
        "Model": name,
        "Accuracy": acc,
        "F1-Score": f1,
        "Precision": precision,
        "Recall": recall,
        "MSE": mse,
        "R2": r2
    }

def main():
    texts, labels = load_data()
    print(f"Loaded {len(texts)} samples")
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM (Linear)": SVC(kernel='linear', probability=True, random_state=42),
        "Naive Bayes": MultinomialNB()
    }
    
    results = []
    for name, clf in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', clf)
        ])
        res = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(res)
        
    df = pd.DataFrame(results)
    
    # Add HuggingFace metrics (previously computed)
    hf_res = {
        "Model": "DistilBERT (Transformers)",
        "Accuracy": 0.8639,
        "F1-Score": 0.8672,
        "Precision": 0.8939,
        "Recall": 0.8421,
        "MSE": 0.1007,
        "R2": 0.5958
    }
    
    df = pd.concat([df, pd.DataFrame([hf_res])], ignore_index=True)
    
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON METRICS")
    print("="*80)
    print(df.to_markdown(index=False, floatfmt=".4f"))

if __name__ == '__main__':
    main()
