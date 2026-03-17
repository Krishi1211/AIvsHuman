import json
import glob
import pandas as pd
import numpy as np
import random
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
    
    ai_texts = []
    
    # Load AI
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if 'sampled' in data:
                    for text in data['sampled']:
                        ai_texts.append(text)
            except Exception as e:
                pass
                
    # Load Humans from Pydriller
    df_human = pd.read_parquet('/Users/krishi1211/Documents/SE/AIvsHuman/Pydriller/commit_metrics.parquet')
    human_texts = df_human['message'].dropna().astype(str).tolist()
    
    ai_count = len(ai_texts)
    random.seed(42)
    human_texts = random.sample(human_texts, ai_count)
    
    texts = human_texts + ai_texts
    labels = [0]*len(human_texts) + [1]*len(ai_texts)
    
    return texts, labels

def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    acc = accuracy_score(y_test, preds)
    mse = mean_squared_error(y_test, probs)
    r2 = r2_score(y_test, probs)
    return {"Model": name, "Accuracy": acc, "F1-Score": f1, "Precision": precision, "Recall": recall, "MSE": mse, "R2": r2}

def main():
    texts, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM (Linear)": SVC(kernel='linear', probability=True, random_state=42),
        "Naive Bayes": MultinomialNB()
    }
    
    results = []
    for name, clf in models.items():
        pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))), ('clf', clf)])
        results.append(evaluate_model(name, pipeline, X_train, X_test, y_train, y_test))
        
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False, floatfmt=".4f"))

if __name__ == '__main__':
    main()
