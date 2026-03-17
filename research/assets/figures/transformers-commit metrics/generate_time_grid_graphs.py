import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("Loading and aggregating datasets...")
usecols_airflow = ['hash', 'author_date', 'insertions', 'deletions', 'files_count']
usecols_elastic = ['hash', 'author_date', 'insertions', 'deletions', 'files']

dfs = []

# Load Airflow Old (2015-2016)
try:
    df1 = pd.read_csv('/Users/krishi1211/Documents/SE/airflow_commits_data_old.csv', usecols=usecols_airflow, on_bad_lines='skip')
    dfs.append(df1)
except Exception as e: print("Skipped old airflow data.")

# Load Elastic 2016-2019
try:
    df2 = pd.read_csv('/Users/krishi1211/Documents/SE/elasticsearch_metrics_2016_2019.csv', usecols=usecols_elastic, on_bad_lines='skip').rename(columns={'files': 'files_count'})
    dfs.append(df2)
except Exception as e: print("Skipped elastic 19 data.")

# Load Airflow modern (~2022)
try:
    df3 = pd.read_csv('/Users/krishi1211/Documents/SE/airflow_commits_data.csv', usecols=usecols_airflow, on_bad_lines='skip')
    dfs.append(df3)
except Exception as e: print("Skipped modern airflow data.")

# Load Elastic 2022-2025
try:
    df4 = pd.read_csv('/Users/krishi1211/Documents/SE/elasticsearch_metrics_2022_2025.csv', usecols=usecols_elastic, on_bad_lines='skip').rename(columns={'files': 'files_count'})
    dfs.append(df4)
except Exception as e: print("Skipped elastic 25 data.")

df = pd.concat(dfs, ignore_index=True)

print("Preprocessing...")
df['author_date'] = pd.to_datetime(df['author_date'], errors='coerce', utc=True)
df = df.dropna(subset=['author_date'])

for col in ['insertions', 'deletions', 'files_count']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\[|\]', '', regex=True), errors='coerce').fillna(0)

# Filter for the relevant years (2015 - 2025)
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

print("Plotting graphs into 2x2 grid with Time on X-Axis...")
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# 1. Commits Over Time (Monthly)
axes[0, 0].plot(monthly_stats.index, monthly_stats['commit_count'], color='#2a9d8f')
axes[0, 0].set_title('1. Monthly Commit Frequency Over Time', fontsize=12)
axes[0, 0].set_xlabel('Time (Year)')
axes[0, 0].set_ylabel('Commit Count')

# 2. Code Churn (Volatility) Over Time
axes[0, 1].plot(monthly_stats.index, monthly_stats['churn'], color='#8b0000')
axes[0, 1].set_title('2. Monthly Code Churn Volatility Over Time', fontsize=12)
axes[0, 1].set_xlabel('Time (Year)')
axes[0, 1].set_ylabel('Total Lines Changed (Insertions + Deletions)')

# 3. Lines Added per Commit (Density Scatter over time)
# We filter out 0 insertions for log scale
insertions_df = commit_df[commit_df['insertions'] > 0]
axes[1, 0].scatter(insertions_df['author_date'], insertions_df['insertions'], alpha=0.15, color='#9fb2cf', s=12, edgecolor='none')
axes[1, 0].set_yscale('log')
axes[1, 0].set_title('3. Lines Added per Commit Over Time (Log Scale)', fontsize=12)
axes[1, 0].set_xlabel('Time (Year)')
axes[1, 0].set_ylabel('Insertions per Commit')

# 4. Files Changed per Commit (Density Scatter over time)
files_df = commit_df[commit_df['files_count'] > 0]
axes[1, 1].scatter(files_df['author_date'], files_df['files_count'], alpha=0.15, color='#ffaa33', s=12, edgecolor='none')
axes[1, 1].set_yscale('log')  # File counts can also spike highly; log helps visualize the spread
axes[1, 1].set_title('4. Files Changed per Commit Over Time (Log Scale)', fontsize=12)
axes[1, 1].set_xlabel('Time (Year)')
axes[1, 1].set_ylabel('Files Changed per Commit')

plt.tight_layout()
output_path = '/Users/krishi1211/Documents/SE/AIvsHuman/research/assets/figures/transformers-commit metrics/repo_analysis_time_grid.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Time-based grid graph successfully generated and saved to {output_path}")
