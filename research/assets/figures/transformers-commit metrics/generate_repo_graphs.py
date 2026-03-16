import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Loading dataset...")
# Load a subset of columns to save memory
usecols = ['hash', 'author_date', 'insertions', 'deletions', 'files_count']
df = pd.read_csv('/Users/krishi1211/Documents/SE/airflow_commits_data_old.csv', usecols=usecols, on_bad_lines='skip')

print("Preprocessing...")
df['author_date'] = pd.to_datetime(df['author_date'], errors='coerce', utc=True)
df = df.dropna(subset=['author_date'])

# Format numbers explicitly
for col in ['insertions', 'deletions', 'files_count']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\[|\]', '', regex=True), errors='coerce').fillna(0)

# Filter for 2015-2024 as requested
df = df[(df['author_date'].dt.year >= 2015) & (df['author_date'].dt.year <= 2024)]

# To get commit level metrics, we can drop duplicate hashes since files_count, insertions, deletions are likely uniform per commit?
# Let's check if insertions/deletions are per commit or per file. In PyDriller, usually, insertions are per commit if extracted at commit level, or per file if extracted at file level.
# We will aggregate by hash: Max of files_count, Sum of insertions and deletions (if it's file-level) 
# Usually, if 'filename' is a column, the dataset is file-level. So insertions might be for the file. Let's sum them per commit.
commit_df = df.groupby('hash').agg({
    'author_date': 'first',
    'insertions': 'sum',
    'deletions': 'sum',
    'files_count': 'max' # Or count how many rows: 'hash': 'count'
}).rename(columns={'files_count': 'max_files_count'})

# Let's actually count rows as files changed as well, just in case max_files_count is broken
file_counts = df.groupby('hash').size()
commit_df['files_changed'] = file_counts

# Calculate code churn (insertions + deletions)
commit_df['churn'] = commit_df['insertions'] + commit_df['deletions']

# Add Year-Month for temporal grouping
commit_df['year_month'] = commit_df['author_date'].dt.to_period('M').dt.to_timestamp()

print(commit_df.head())

# 1. Commits Over Time (Monthly) & 2. Code Churn (Volatility)
monthly_stats = commit_df.groupby('year_month').agg({
    'author_date': 'count', # Number of commits
    'churn': 'sum' # Total churn
}).rename(columns={'author_date': 'commit_count'})

# Plotting Set 1 (Commits over time & Churn)
plt.style.use('ggplot')
sns.set_palette("husl")
fig, ax1 = plt.subplots(figsize=(14, 6))

color = 'tab:blue'
ax1.set_xlabel('Date (Monthly)')
ax1.set_ylabel('Commit Frequency', color=color)
ax1.plot(monthly_stats.index, monthly_stats['commit_count'], color=color, label='Commits', alpha=0.9, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Code Churn (Lines Added + Deleted)', color=color)
ax2.plot(monthly_stats.index, monthly_stats['churn'], color=color, label='Code Churn', alpha=0.5, linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('Monthly Commit Frequency and Code Churn in Apache Airflow (2015-2024)', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('/Users/krishi1211/Documents/SE/monthly_commits_and_churn.png', dpi=300)
plt.close()

# 3. Lines Added Distribution (Log Scale)
plt.figure(figsize=(10, 6))
# Filter out 0 to allow log scale
insertions = commit_df[commit_df['insertions'] > 0]['insertions']
# We plot standard histogram but with log-scaled bins and log y-axis
bins = np.logspace(np.log10(1), np.log10(insertions.max()), 50)
sns.histplot(insertions, bins=bins, color='seagreen', kde=False)
plt.xscale('log')
plt.yscale('log')
plt.title('Lines Added Distribution (Log Scale)', fontsize=14, pad=15)
plt.xlabel('Number of Lines Added per Commit (Log Scale)')
plt.ylabel('Frequency (Log Scale)')
plt.tight_layout()
plt.savefig('/Users/krishi1211/Documents/SE/lines_added_distribution.png', dpi=300)
plt.close()

# 4. Files Changed Distribution
plt.figure(figsize=(10, 6))
files_changed = commit_df['files_changed']
# Limit the heavily right skewed graph to 99th percentile for better visibility
limit = files_changed.quantile(0.99)
filtered_files = files_changed[files_changed <= limit]
sns.histplot(filtered_files, binwidth=1, color='purple', discrete=True)
plt.title('Files Changed per Commit Distribution (Capped at 99th Percentile)', fontsize=14, pad=15)
plt.xlabel('Number of Files Changed')
plt.ylabel('Frequency (Number of Commits)')
plt.xlim(0, limit)
plt.tight_layout()
plt.savefig('/Users/krishi1211/Documents/SE/files_changed_distribution.png', dpi=300)
plt.close()

print("Graphs generated successfully.")
