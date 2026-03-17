# 🚀 AIvsHuman Execution Logic Workflow

Follow this step-by-step pipeline to extract fresh project metadata out of target repositories, train the `distilbert` classification array optimally over that structural footprint, and reconstruct the timeline metrics revealing exactly when developers inside that particular organization began utilizing LLMs for auto-generated outputs.

---

### Step 1: Raw Ground-Truth Extraction
Begin by targeting major open source frameworks to serve as your pure `Human (0)` experimental control strings.

1. Navigate to the `Pydriller/` directory.
2. Edit `extract_airflow_data.py` (or duplicate it for your own framework). Map out the exact Git URL endpoint you wish to audit, and the chronological years you wish to scrape:
   ```python
   repo_url = "https://github.com/YourTarget/yourFramework"
   start_date = datetime(2015, 1, 1)
   end_date = datetime(2025, 1, 1)
   ```
3. Run `python extract_airflow_data.py`. Expect heavy processing loops. PyDriller will clone the target branch internally and mathematically trace every single insertion, deletion, code file change, and commit message logging chronologies recursively. 
4. Confirm large outputs (like `1GB+` scale `.csv` dumps) successfully dropped directly inside the local `Pydriller` directory (the `.gitignore` shields your repository natively). Ensure your pipeline converted the relevant strings natively to `commit_metrics.parquet` datasets for maximum memory efficiency loading arrays.

---

### Step 2: Training The Classification Array
Once the structural control data exists locally, transition directories into your experimental test environment.

1. Navigate to the `transformer-commit message/` directory.
2. Ensure you have the datasets generated. The training algorithms intrinsically sample the generated `Pydriller/commit_metrics.parquet` strings mapping historical human phrasing and juxtaposes them directly inside arrays against `detect-gpt-fork` generic AI LLM sentences.
3. Establish your HuggingFace Transformer deep-learning environment:
   ```bash
   pip install transformers datasets torch scikit-learn
   ```
4. Fine-Tune the HuggingFace Machine Learning Array:
   ```bash
   python codes/train_ai_detector_hf.py --train
   ```
   **Output:** The script dynamically splits validation sets, processes thousands of tokens using `distilbert-base-uncased` across MPS/CUDA arrays, and dynamically rewrites a finalized, mathematically optimized logic `.safetensors` binary directly into the `.saved_hf_detector/` module.

*(Optional Control Test): Validate classic machine learning algorithms locally to contrast and prove the HuggingFace logic metrics natively.*
```bash
python codes/train_ai_detector_multi.py
```

---

### Step 3: Visualization & Chronological Predictions
With a calibrated LLM model natively checking terse phrasing probabilities accurately, we trace it back into history!

1. Start analyzing the real distributions to chart exactly when AI began rewriting Open Source rules. Run the time-series trajectory algorithms parsing the unlabelled PyDriller CSV exports:
   ```bash
   # Automatically executes AI probability traces over thousands of sequential dates:
   python codes/plot_ai_usage.py
   ```
   **Output:** The system dumps exact density tables natively onto standard out, logging yearly statistical adoptions mapping out ChatGPT API and Copilot expansion curves locally! The chronological visual graph is dropped inside `images/ai_usage_trend.png`.

2. To map massive volatility over time, density-scaling behaviors, and code refactor impacts—which might influence misclassification on older dates—simply process your dataset density matrixes:
   ```bash
   python codes/generate_time_grid_graphs.py
   ```
   **Output**: A finalized multi-chart data-board written visually directly to `images/repo_analysis_time_grid.png`. 

Congratulations! You've accurately traced the hidden footprints of modern generative developers inside major Open Source frameworks.
