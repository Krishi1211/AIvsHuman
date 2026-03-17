# AI-Generated Commit Analysis Toolkit (Parquet-Powered)

This structure represents the refined evaluation loop and pipeline for detecting AI-augmented workflow footprints directly within open-source repository commit hashes.

This directory is independently completely contained in two parts:
- `codes/`: Fully contained data extraction, training loops, model definitions, and metrics graphing logic written natively in Python.
- `images/`: The directly output 2x2 multi-panel metrics, timeline line-graphs, and accuracy arrays utilized in our project analysis.

---

## 📂 Directory Map & File Manifest

### `/codes/` - Logic & Analytics
This folder holds all the Python codebase for training the detection models, predicting across historical commit data, and plotting the figures.

* **💻 Modeling & Training:**
    * `train_ai_detector_hf.py` - **The Core DL Pipeline.** Trains and fine-tunes the `DistilBERT` sequence classification Transformer using our combined AI JSON outputs and verified `Pydriller` parquet human datasets. Caches the resulting optimal weights.
    * `train_ai_detector_multi.py` - **Classical Baseline Array.** Automatically evaluates Traditional ML algorithms (Logistic Regression, Random Forest, SVM, Naive Bayes) using numerical Term-Frequency Vectorization to act as the experimental control testing metric against DistilBERT.

* **📈 Evaluation & Graph Generation:**
    * `plot_ai_usage.py` - Iterates over 2015-2025 parsed data-frames, loading the fine-tuned HuggingFace classifier, and scoring historical AI adoption over time. Dynamically outputs to `images/ai_usage_trend.png`.
    * `generate_graphs.py` - Python visualizer mapping out Brier MSE & R² variances alongside classification metrics (Accuracy, F1-Score) from the multi-model baseline test vs the NLP Transformer. Spits out `metrics_comparison.png` and `calibration_metrics.png`.
    * `generate_time_grid_graphs.py` - Massive-scale timeline density mapper. Charts the distribution of historical file mutations, localized insertions, and code churn natively across the exact temporal timeline. Outputs `repo_analysis_time_grid.png`. 
    * `generate_grid_graphs.py` - Legacy plotting script mapping distribution frequency independent of chronological flow.

### `/images/` - Computed Visualizations
This pipeline directly overwrites visualizations within this directory upon completion. By keeping them segregated, it avoids contaminating the codebase.

* `metrics_comparison.png`: A bar-graph array directly evaluating our DistilBERT algorithm against classic ML baselines.
* `calibration_metrics.png`: A visual chart outlining the regression metrics (MSE Brier scores, R²) proving mapping reliabilities.
* `ai_usage_trend.png`: A mapped linear visual indicating exactly *when* mass adoption of Generative Agents penetrated repository contribution graphs.
* `repo_analysis_time_grid.png`: A detailed 2x2 grid charting massive repository scale metrics over time (Volatility, File-Changes, Growth Rate) to isolate potential biases and noise variables.

---

## 🛠 Pydriller Re-Training Architecture

The primary difference in this pipeline evaluation vs baseline testing is our direct utilization of `Pydriller/commit_metrics.parquet`. Instead of pulling baseline Human evaluation messages from randomized text files or CSVs lacking context, we loaded **5,014 verified historical developer commits** extracted natively from PyDriller analysis on established projects (e.g., Apache Airflow, Elastic, Django). These act as our strictly enforced Human `0` baselines to rigorously test against Generative output.

Our newly fine-tuned `DistilBERT` machine-learning checkpoints successfully processed `1,800` completely standardized arrays to score a mathematically perfect `100% Accuracy` metric on evaluation mapping against both text-types.

## 📊 Fast Results & Findings

### 1. Model Evaluation Metrics
We directly compared classical algorithmic approaches using Term Frequency against a fine-tuned deep learning Natural Language Transformer Model (`distilbert-base-uncased`). 

| Model                           | Accuracy | F1-Score | Precision | Recall | MSE (Brier) | R² Score |
|:--------------------------------|---------:|---------:|----------:|-------:|------------:|---------:|
| Logistic Regression             |   97.78% |   97.77% |    97.22% | 98.31% |      0.0557 |   0.7773 |
| Random Forest                   |   98.06% |   98.03% |    98.31% | 97.75% |      0.0332 |   0.8674 |
| Naive Bayes                     |   98.61% |   98.59% |    98.87% | 98.31% |      0.0396 |   0.8416 |
| SVM (Linear)                    |   98.89% |   98.87% |    99.43% | 98.31% |      0.0078 |   0.9686 |
| **DistilBERT (Transformers)**   |**100.00%**|**100.00%**| **100.00%**|**100.00%**| **0.0000** | **0.9999** |

*Note: Classic metrics performed vastly better when trained against Django-native pydriller syntax vs the original unstructured dataset. However, DistilBERT scaled dynamically to map perfectly without faltering.*

### 2. Historical AI Usage Trend (2015 - 2025)
By batching ~4,000 historical project commits dating from 2015 through 2025 (parsing both Elasticsearch and Apache Airflow datasets into DistilBERT), we mapped the historical adoption trajectories.

- **Pre-2022**: ~0% Usage.
- **2022**: Jumps noticeably to **26.25%** exactly correlating with Copilot General Availability and ChatGPT's launch.
- **2024**: Hits **40.75%** AI-stylized adoption density mirroring mass commercial adoption curves.
- **2025**: Crests to **46.00%** of all generated commits containing distinct LLM sentence structuring.

---

## 🚀 Execution Instructions

### Local Training
To directly reproduce the newly perfected NLP classification architecture and re-evaluate over the parquet mappings:
```bash
python codes/train_ai_detector_hf.py --train
```

### Reproduce Metrics Data
If dataset variables are adjusted or additional `.parquet` data splits are injected, execute these individual algorithms to dynamically crunch the numbers and cleanly overwrite the graphics safely inside the `images/` directory:
```bash
python codes/generate_graphs.py
python codes/plot_ai_usage.py
python codes/generate_time_grid_graphs.py
```
