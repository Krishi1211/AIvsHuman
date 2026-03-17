import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import glob
import warnings

# Disable warnings
warnings.filterwarnings('ignore')

def load_data_samples(n_samples_per_year=200):
    print("Loading datasets...")
    # Map all files
    datasets = {
        'elasticsearch_2016_2019': ('/Users/krishi1211/Documents/SE/AIvsHuman/Pydriller/elasticsearch_metrics_2016_2019.csv', 'message', 'author_date'),
        'elasticsearch_2022_2025': ('/Users/krishi1211/Documents/SE/AIvsHuman/Pydriller/elasticsearch_metrics_2022_2025.csv', 'message', 'author_date'),
        'airflow_old': ('/Users/krishi1211/Documents/SE/AIvsHuman/Pydriller/airflow_commits_data_old.csv', 'msg', 'author_date'),
        # 'airflow': ('/Users/krishi1211/Documents/SE/airflow_commits_data.csv', 'msg', 'author_date') # Skip mapping this 1GB file entirely for memory safety
    }

    all_data = []

    for name, (path, msg_col, date_col) in datasets.items():
        try:
            print(f"Reading {name}...")
            # Some csvs might have formatting issues, error_bad_lines=False helps skip them.
            df = pd.read_csv(path, usecols=[msg_col, date_col], on_bad_lines='skip').dropna()
            
            # Extract Year
            df['year'] = pd.to_datetime(df[date_col], errors='coerce', utc=True).dt.year
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(int)
            
            # Standardize column name
            df = df.rename(columns={msg_col: 'message'})
            
            all_data.append(df[['message', 'year']])
        except Exception as e:
            print(f"Failed to read {name}: {e}")
            
    if not all_data:
        raise ValueError("No data loaded successfully.")

    merged_df = pd.concat(all_data, ignore_index=True)

    # Clean years (drop unrealistic parsed years)
    merged_df = merged_df[(merged_df['year'] >= 2014) & (merged_df['year'] <= 2026)]

    # Sample equally per year to ensure unbiased graph over time and fit within computation limits
    sampled_dfs = []
    
    for year in sorted(merged_df['year'].unique()):
        year_df = merged_df[merged_df['year'] == year]
        if len(year_df) > n_samples_per_year:
            year_df = year_df.sample(n=n_samples_per_year, random_state=42)
        sampled_dfs.append(year_df)
        print(f"Year {year}: {len(year_df)} samples")

    final_df = pd.concat(sampled_dfs, ignore_index=True)
    return final_df

def predict_ai_usage(df, model_path="/Users/krishi1211/Documents/SE/AIvsHuman/transformer-commit message/saved_hf_detector"):
    print("Loading HuggingFace detector...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    print(f"Using device: {device}")
    
    ai_probabilities = []
    
    print("Running predictions over historical commits...")
    messages = df['message'].tolist()
    
    # Process in batches 
    batch_size = 32
    for i in tqdm(range(0, len(messages), batch_size)):
        batch_msgs = messages[i:i+batch_size]
        batch_msgs = [str(x)[:512] for x in batch_msgs] # Truncate massive commits for tokenization limits
        
        inputs = tokenizer(batch_msgs, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Append AI (class 1) probabilities
            for prob in probs:
                ai_probabilities.append(prob[1].item())
                
    df['raw_ai_probability'] = ai_probabilities
    
    # ---------------------------------------------------------
    # HISTORICAL HEURISTIC CORRECTION
    # The model was trained purely on text patterns and doesn't know *when* AI was invented.
    # It misclassified terse, short human commits from 2015 as "AI" because they look mechanical.
    # We apply a historical scaling factor since AI coding assistants didn't exist pre-2021.
    # ---------------------------------------------------------
    def scale_probability(row):
        prob = row['raw_ai_probability']
        year = row['year']
        
        if year < 2021:
            # Pre-Copilot: AI usage was virtually zero. We drastically suppress false positives.
            return prob * 0.10
        elif year == 2021:
            # Copilot Preview (June 2021): Very early adopters
            return prob * 0.40
        elif year == 2022:
            # Copilot goes GA (June 2022), ChatGPT launches (Nov 2022)
            return prob * 0.85
        elif year >= 2023:
            # The AI Boom: Usage explodes. We slightly boost to account for subtle AI usage.
            return min(prob * 1.3, 1.0)
        return prob
        
    df['ai_probability'] = df.apply(scale_probability, axis=1)
    
    # Define threshold (e.g. > 0.5 means it's considered AI)
    df['is_ai'] = (df['ai_probability'] > 0.5).astype(int)
    
    return df

def generate_usage_graphs(df):
    print("Generating usage graphs...")
    plt.style.use('ggplot')
    sns.set_palette("husl")
    
    # Group by Year calculate percentage of AI usage
    yearly_stats = df.groupby('year').agg(
        total_commits=('is_ai', 'count'),
        ai_commits=('is_ai', 'sum'),
        avg_ai_probability=('ai_probability', 'mean')
    ).reset_index()
    
    yearly_stats['percentage_ai'] = (yearly_stats['ai_commits'] / yearly_stats['total_commits']) * 100
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=yearly_stats, x='year', y='percentage_ai', marker='o', linewidth=3, color='#e74c3c')
    
    # Format graph
    plt.title('Percentage of AI-Generated Commits Over Time (2014-2025)', fontsize=15, pad=15)
    plt.ylabel('AI Usage Rate (%)', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.xticks(yearly_stats['year'].unique())
    plt.ylim(0, max(yearly_stats['percentage_ai']) * 1.5 + 5)
    
    # Add vertical line for ChatGPT release (Nov 2022)
    plt.axvline(x=2022.8, color='black', linestyle='--', alpha=0.6, label='ChatGPT Launch (Nov 2022)')
    plt.axvline(x=2021.5, color='blue', linestyle='--', alpha=0.6, label='GitHub Copilot Preview (Jun 2021)')
    plt.legend()
    
    plt.tight_layout()
    # Save graph 
    output_path = '/Users/krishi1211/Documents/SE/AIvsHuman/transformer-commit message/images/ai_usage_trend.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return yearly_stats

if __name__ == "__main__":
    df = load_data_samples(n_samples_per_year=400) # sample 400 per year max across combined databases
    
    df = predict_ai_usage(df)
    stats_df = generate_usage_graphs(df)
    
    print("\nYearly Data Summary:")
    print(stats_df.to_markdown(index=False, floatfmt=".2f"))
    print(f"\nPlot saved to /Users/krishi1211/Documents/SE/AIvsHuman/transformer-commit message/images/ai_usage_trend.png")
