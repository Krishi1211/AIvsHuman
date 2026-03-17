# Repository Directory Overview

This `AIvsHuman` repository functions dynamically in separate operational environments to avoid data-contamination across its Gigabyte-level datasets. To utilize the toolkits correctly, it's vital to know where scripts belong.

Below is an overview of the critical root-level directories.

---

### `📁 Pydriller/`
**Purpose**: Big-Data Source Extraction & Human Control Dataset Validation.
This folder acts as the local data-lake. Scripts native here (like `extract_airflow_data.py`) clone massive target open-source repositories directly natively over Git and map every single file addition, deletion, text string, and timestamp from developers across history.

Crucially, **PyDriller arrays act as our `0` (Human) baseline variables for Machine Learning.** 
Data structures (`.json`, `.csv`, `.parquet`) generated out of this module regularly eclipse `1GB+` in size natively, and are strictly ignored via `.gitignore` to prevent crashing GitHub remote tracking limits.

---

### `📁 transformer-commit message/`
**Purpose**: Logic Evaluation, Pipeline Execution, & Generated Visualization Storage.
This is the machine learning engine-room. It uses an active deep-learning LLM (specifically a `distilbert` Transformer model configured by HuggingFace) to parse the human datasets directly from `Pydriller` and score their similarity or trace-signatures compared against ChatGPT models.
- **`codes/`**: Contains Python ML architectures (`train_ai_detector_hf.py`), validation control scripts testing classical implementations (`train_ai_detector_multi.py`), and visualization chart builders mapping AI probabilities onto chronological timelines.
- **`images/`**: Directly receives the automatically generated `.png` graphs summarizing regression stats and temporal ML adoption timelines mapping back 10+ years across GitHub scale.
- **`saved_hf_detector/`**: Holds optimized logic weights dynamically cached after successful full-dataset evaluation rounds.

---

### `📁 detect-gpt-fork/`
**Purpose**: AI Baseline Construction Arrays.
Originally a branch of standard generative detection research, this folder stores baseline `JSON` output strings constructed dynamically from modern Large Language Models (LLMs) used to train the `.1` side of the prediction classifier, enabling the network to establish stylistic boundaries when measuring against our Human text records.

---

### `📁 GPTSnifferInferenceApproach/`
**Purpose**: Research Pipelines (Legacy Testing).
Contains overarching theoretical pipeline testing files analyzing stylistic behaviors within long-form coding functions explicitly, distinct from the specialized terse string analysis located natively inside `transformer-commit message/`.
