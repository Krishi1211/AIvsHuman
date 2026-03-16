import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for modern looking plots
plt.style.use('ggplot')
sns.set_palette("husl")

# Data from our evaluation
data = {
    'Model': ['Naive Bayes', 'SVM (Linear)', 'Logistic Regression', 'Random Forest', 'DistilBERT'],
    'Accuracy': [0.2444, 0.2861, 0.3111, 0.4944, 0.8639],
    'F1-Score': [0.2486, 0.3110, 0.3224, 0.5081, 0.8672],
    'Precision': [0.2616, 0.3169, 0.3352, 0.5222, 0.8939],
    'Recall': [0.2368, 0.3053, 0.3105, 0.4947, 0.8421],
    'MSE': [0.3369, 0.2105, 0.2936, 0.2974, 0.1007],
    'R2_Score': [-0.3518, 0.1554, -0.1782, -0.1933, 0.5958]
}

df = pd.DataFrame(data)

def plot_classification_metrics():
    """Bar chart comparing Accuracy, F1, Precision, and Recall."""
    plt.figure(figsize=(12, 6))
    
    # Melt the dataframe for seaborn grouped barplot
    metrics_df = df[['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall']]
    melted_df = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Score')
    
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=melted_df)
    plt.title('Classification Metrics by Model (AI vs Human Commits)', fontsize=14, pad=15)
    plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
    plt.xlabel('', fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('/Users/krishi1211/Documents/SE/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration_metrics():
    """Bar chart comparing MSE and R2 Scores."""
    plt.figure(figsize=(12, 5))
    
    # Create two subplots side-by-side for MSE and R2 since their scales are very different
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # MSE Plot (Lower is better)
    sns.barplot(x='Model', y='MSE', data=df, ax=ax1, palette='mako')
    ax1.set_title('Mean Squared Error (Brier Score)\n*Lower is Better*', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.set_ylabel('MSE Error')
    
    # R2 Plot (Higher is better, can be negative)
    sns.barplot(x='Model', y='R2_Score', data=df, ax=ax2, palette='rocket')
    ax2.set_title('R² (Explained Variance vs Random Guess)\n*Higher is Better*', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.set_ylabel('R² Score')
    ax2.axhline(0, color='black', linewidth=1) # Draw line at 0 for random guessing baseline
    
    plt.tight_layout()
    plt.savefig('/Users/krishi1211/Documents/SE/calibration_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating graphs...")
    plot_classification_metrics()
    plot_calibration_metrics()
    print("Graphs saved to /Users/krishi1211/Documents/SE/")
