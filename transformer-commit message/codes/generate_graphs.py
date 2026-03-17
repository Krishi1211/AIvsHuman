import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data from our NEW evaluation using Parquet
data = {
    'Model': ['Naive Bayes', 'SVM (Linear)', 'Logistic Regression', 'Random Forest', 'DistilBERT'],
    'Accuracy': [0.9861, 0.9889, 0.9778, 0.9806, 1.000],
    'F1-Score': [0.9859, 0.9887, 0.9777, 0.9803, 1.000],
    'Precision': [0.9887, 0.9943, 0.9722, 0.9831, 1.000],
    'Recall': [0.9831, 0.9831, 0.9831, 0.9775, 1.000],
    'MSE': [0.0396, 0.0078, 0.0557, 0.0332, 0.000002],
    'R2_Score': [0.8416, 0.9686, 0.7773, 0.8674, 0.9999]
}

df = pd.DataFrame(data)

def plot_classification_metrics():
    plt.figure(figsize=(12, 6))
    metrics_df = df[['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall']]
    melted_df = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Score')
    sns.barplot(x='Model', y='Score', hue='Metric', data=melted_df)
    plt.title('Classification Metrics by Model (AI vs Human Commits - Parquet)', fontsize=14, pad=15)
    plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
    plt.ylim(0.8, 1.05)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('/Users/krishi1211/Documents/SE/AIvsHuman/transformer-commit message/images/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration_metrics():
    plt.figure(figsize=(12, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x='Model', y='MSE', data=df, ax=ax1, palette='mako')
    ax1.set_title('Mean Squared Error (Brier Score)\n*Lower is Better*', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    sns.barplot(x='Model', y='R2_Score', data=df, ax=ax2, palette='rocket')
    ax2.set_title('R² (Explained Variance vs Random Guess)\n*Higher is Better*', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig('/Users/krishi1211/Documents/SE/AIvsHuman/transformer-commit message/images/calibration_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_classification_metrics()
    plot_calibration_metrics()
