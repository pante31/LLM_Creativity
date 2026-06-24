# Demystifying Automatic Creativity Evaluation in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Official code and dataset repository for the research paper:** *Demystifying automatic creativity evaluation in large language models*
**Authors:** Alessandro Tutone, Giorgio Franceschelli, Mirco Musolesi

---

## 📖 Overview

As Large Language Models (LLMs) achieve remarkable proficiency in natural language generation, accurately measuring their capacity for creativity remains a profound academic challenge. This repository contains the official code, datasets, and experimental framework designed to investigate whether current automatic evaluation schemes possess the capacity to correctly evaluate human and artificial creativity.

Using a curated subset of the *WritingPrompts* dataset (comprising both human-authored and LLM-generated short stories), this project conducts a multi-dimensional comparative analysis. It contrasts objective statistical metrics and the "LLM-as-a-Judge" paradigm against a robust baseline of human subjective evaluations across 11 distinct dimensions of creativity.

## 🚀 Key Findings

* **Inadequacy of Automated Metrics:** Traditional mathematical text evaluation frameworks (including the Creativity Index, Perplexity, and syntactic template scores) exhibit near-zero alignment with human perception of aesthetic quality, surprise, and emotional resonance.
* **Severe Algorithmic Bias:** The LLM-as-a-Judge paradigm is heavily driven by a systemic self-preference bias, consistently favoring the probabilistic, low-variance stylistic signatures of AI-generated texts over human unpredictability.
* **Semantic Misalignment:** Correlation analyses reveal that human evaluators strongly associate creativity with novelty, originality, and surprise. In contrast, LLMs primarily correlate creativity with surface-level structural elaboration.
* **The Risk of an Echo Chamber:** Unsupervised reliance on automated evaluation for creative tasks risks establishing an algorithmic echo chamber, contributing to model collapse and standardizing a sterile definition of creativity.

## 🛠️ Methodology & Experimental Design

* **Dataset:** A balanced corpus of 200 short stories (100 human-written sourced from the *WritingPrompts* collection, and 100 AI-generated using state-of-the-art models including GPT-5.2, DeepSeek-V3.2, Mistral Large 3, Claude Sonnet 4.5, and Gemini 3 Pro).
* **Automated Metrics:** Perplexity, syntactic template analysis (TR, TPT, CR-POS), Expectation-Adjusted Distinct n-grams (EAD), Semantic Diversity (SBERT-Div), and the Creativity Index.
* **Subjective Evaluation:** A dual-track blind evaluation protocol comparing human-in-the-loop judgments against an isolated LLM-as-a-Judge framework across 11 cognitive and aesthetic dimensions (e.g., Authenticity, Effectiveness, Elaboration, Novelty, Surprise).

## 📂 Repository Structure

```text
📂 LLM_Creativity
├── 📁 app          # Web interface code used for the human evaluation campaign
├── 📁 dataset      # Raw and processed WritingPrompts subset (Human & AI stories)
├── 📁 imgs         # High-resolution figures and correlation heatmaps from the paper
├── 📁 metrics      # Core logic for Automated Metrics & LLM-as-a-Judge inference
├── 📁 notebooks    # Jupyter notebooks for statistical analysis and visualization
├── 📁 results      # Final outputs, p-values, and correlation matrices
├── 📄 .gitignore   # Git configurations
└── 📄 README.md    # Project documentation
```

## ⚙️ Installation & Usage

To ensure full reproducibility of the paper's findings, or to apply the evaluation pipeline to your own text datasets:

1. Clone the repository:
   ```bash
   git clone [https://github.com/pante31/LLM_Creativity.git](https://github.com/pante31/LLM_Creativity.git)
   cd LLM_Creativity
   cd metrics
   ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script to compute all automated metrics:
    ```bash
    python3 creativity.py
    ```
**💡 Note on Evaluation Logic:** The ``creativity.py`` script serves as the core engine of this repository. It contains all the necessary logic, algorithms, and prompt templates required for text evaluation. This includes the implementation of the objective automated metrics (e.g., Perplexity, POS tagging extraction) as well as the exact, isolated prompt engineering instructions used to query the LLM-as-a-Judge.

## 📊 Data Availability

All generated and collected data used in this study, including the anonymized human survey responses and the LLM inference outputs, are publicly available within the ``dataset`` and ``results`` directories of this repository. The original prompts and human-written short stories are sourced from the public *WritingPrompts* dataset, available via HuggingFace.

## 📄 Citation

If you utilize this code, dataset, or experimental framework in your own research, please cite our paper:

  ```
   @article{tutone2026demystifying,
      title        = {Demystifying automatic creativity evaluation in large language models},
      author       = {Tutone, Alessandro and Franceschelli, Giorgio and Musolesi, Mirco},
      journal      = {Nature Machine Intelligence},
      note         = {Under Review},
      year         = {2026}
   }
  ```


*"Rest at the End, Not in the Middle"*
