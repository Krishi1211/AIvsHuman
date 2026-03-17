import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("Loading dataset...")
usecols = ['hash', 'author_date', 'insertions', 'deletions', 'files_count']
df = pd.read_csv('/Users/krishi1211/Documents/SE/airflow_commits_data_old.csv', usecols=usecols, on_bad_lines='skip')

print("Preprocessing...")
df['author_date'] = pd.to_datetime(df['author_date'], errors='coerce', utc=True)
df = df.dropna(subset=['author_date'])

# Format columns safely
for col in ['insertions', 'deletions', 'files_count']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\[|\]', '', regex=True), errors='coerce').fillna(0)

# Filter for the relevant years (e.g., 2015 - 2025 as requested)
df = df[(df['author_date'].dt.year >= 2015) & (df['author_date'].dt.year <= 2025)]

print("Aggregating metrics per commit...")
commit_df = df.groupby('hash').agg({
    'author_date': 'first',
    'insertions': 'sum',
    'deletions': 'sum',
    'files_count': 'max'
})

commit_df['churn'] = commit_df['insertions'] + commit_df['deletions']
commit_df['month_year'] = commit_df['author_date'].dt.to_period('M').dt.to_timestamp()

monthly_stats = commit_df.groupby('month_year').agg({
    'author_date': 'count',
    'churn': 'sum'
}).rename(columns={'author_date': 'commit_count'})

print("Plotting graphs into 2x2 grid...")
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# 1. Commits Over Time (Monthly)
axes[0, 0].plot(monthly_stats.index, monthly_stats['commit_count'], color='#2a9d8f')
axes[0, 0].set_title('1. Commits Over Time (Monthly)', fontsize=12)
axes[0, 0].set_xlabel('month_year')
axes[0, 0].set_ylabel('')

# 2. Code Churn (Volatility)
axes[0, 1].plot(monthly_stats.index, monthly_stats['churn'], color='#8b0000')
axes[0, 1].set_title('2. Code Churn (Volatility)', fontsize=12)
axes[0, 1].set_xlabel('month_year')
axes[0, 1].set_ylabel('')

# 3. Lines Added Distribution (Log Scale)
insertions = commit_df[commit_df['insertions'] > 0]['insertions']
sns.histplot(insertions, bins=50, log_scale=(True, False), ax=axes[1, 0], color='#9fb2cf', edgecolor='white')
axes[1, 0].set_title('3. Lines Added Distribution (Log Scale)', fontsize=12)
axes[1, 0].set_xlabel('insertions')
axes[1, 0].set_ylabel('Count')

# 4. Files Changed Distribution
files_changed = commit_df[commit_df['files_count'] > 0]['files_count']
sns.histplot(files_changed, bins=30, log_scale=(True, False), ax=axes[1, 1], color='#ffaa33', edgecolor='white')
axes[1, 1].set_title('4. Files Changed Distribution', fontsize=12)
axes[1, 1].set_xlabel('files')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
output_path = '/Users/krishi1211/Documents/SE/repo_analysis_grid.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Grid graph successfully generated and saved to {output_path}")
