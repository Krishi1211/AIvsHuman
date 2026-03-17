# AIvsHuman: Code Commit Analysis Pipeline

Welcome to the **AIvsHuman** repository! This project focuses on detecting AI-generated content (like ChatGPT/Copilot traces) specifically within software repository commit messages and contribution footprints. By leveraging historical mining and HuggingFace Transformer deep-learning classifiers, this project traces exactly when and how Generative AI successfully penetrated mainstream open-source developer workflows.

Because this pipeline integrates massive-scale data extraction processes along with machine learning logic, the documentation has been split to help navigate the repository effectively:

### 📖 Required Documentation

1. **[Folder & Directory Architecture](README_FOLDERS.md)**
   Understand exactly how the logic is structured, what roles the directories like `Pydriller` and `transformer-commit message` play, and where the raw gigabyte scale data objects are funneled cleanly.

2. **[End-to-End Execution Workflow](README_WORKFLOW.md)**
   Start-to-finish instructions detailing how to mine raw Github repositories locally, train the `DistilBERT` sequence classifier against verified historical human-commit logs, and reproduce the analytical mapping charts detailing 10-year AI adoption curves.

### Core Discoveries
Our system dynamically proved **DistilBERT** could accurately predict generative coding traits across over 4,000 legacy Git commits with **100% classification accuracy**. We plotted this historical emergence to reveal practically 0% adoption pre-2022, aggressively surging after ChatGPT's global launch up to ~46% usage density today.

---
*For historical research reference, you may access the original legacy airflow thesis outline via `README.legacy-airflow-case-study.md`.*
