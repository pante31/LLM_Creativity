# A Critical Analysis of LLM Creativity Evaluation 🧠✍️

**Author:** Alessandro Tutone  
**Institution:** Alma Mater Studiorum - University of Bologna (Master's Degree in Artificial Intelligence)  
**Thesis Supervisor:** Prof. Mirco Musolesi  
**Thesis Tutor:** Giorgio Franceschelli  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 Overview

Large Language Models (LLMs) have achieved remarkable proficiency in natural language generation, increasingly challenging human benchmarks in abstract domains such as creative writing. However, accurately measuring creativity and understanding how these models relate to such a complex construct remains a profound academic challenge. 

This repository contains the code, data analysis, and experimental framework for my Master's thesis, which investigates the fundamental relationship between LLMs and creativity. It asks not only whether AI can successfully simulate creative works, but also **whether LLMs possess the capacity to objectively evaluate them.**

Using the *WritingPrompts* dataset, this project conducts a multi-dimensional comparative analysis of human-authored and LLM-generated narratives across phrase-level lexical choices, structural templates, and 11 subjective dimensions of creativity.

## 🚀 Key Findings

1. **The LLM-as-a-Judge Bias & The Echo Chamber:** The study exposes a severe systemic bias within the LLM-as-a-judge paradigm. The AI evaluator consistently favored its own probabilistic, low-variance stylistic signatures, entirely failing to recognize the narrative unpredictability and organic prosody valued by human readers. Unsupervised reliance on this assessment risks establishing an "algorithmic echo chamber" and contributing to model collapse.
2. **The Quality-Diversity Pareto Front:** Principal Component Analysis (PCA) revealed a stark topological divide. AI-generated texts are rigidly constrained along a mathematical quality-diversity Pareto frontier. In contrast, human narratives exhibit high variance, effortlessly breaking these algorithmic boundaries.
3. **The Failure of Objective Metrics:** Correlation analyses demonstrated that current mathematical evaluation frameworks are fundamentally inadequate. Purely structural metrics, such as the *Creativity Index*, exhibited near-zero alignment with human perception of aesthetic quality, surprise, and emotional resonance.

## 🛠️ Methodology & Experimental Design

* **Dataset:** Subset of the [*WritingPrompts*](https://arxiv.org/abs/1805.04833) collection.
* **Automated Metrics:** Perplexity, syntactic template analysis (TPT, EAD, CR-POS), and the Creativity Index.
* **Subjective Evaluation:** Human-in-the-loop evaluation vs. LLM-as-a-judge framework across 11 cognitive and aesthetic dimensions (including Novelty, Originality, Value, Surprise, and Effectiveness, inspired by Boden and Runco).

## 📂 Repository Structure

```text
LLM_Creativity/
│
├── app/                    # Source code for the human evaluation web interface
├── dataset/                # Raw and processed datasets (WritingPrompts subset)
├── imgs/                   # Images, plots, and figures generated during analysis
├── metrics/                # Python scripts for automated metric computation (Perplexity, POS, etc.)
├── notebooks/              # Jupyter notebooks for PCA, Correlation, Statistical Analysis, and other tests
├── results/                # Output data, generated evaluations, and correlation matrices
├── .gitignore              # Git ignore configurations
└── README.md               # Project documentation
```

## ⚙️ Installation & Usage

To replicate the correlation matrices or run the automated metrics on your own text data:

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

## 📄 Citation

If you use this code, dataset, or research in your own work, please cite the thesis:

  ```@mastersthesis{tutone2026llmcreativity,
    author       = {Alessandro Tutone},
    title        = {A Critical Analysis of LLM Creativity Evaluation},
    school       = {Alma Mater Studiorum - University of Bologna},
    year         = {2026},
    type         = {Master's Thesis},
    note         = {Department of Computer Science and Engineering (DISI)}
  }
  ```

## 🤝 Acknowledgments

Special thanks to my family for their unwavering support, to my supervisor Prof. Mirco Musolesi, and to my tutor Giorgio Franceschelli for their invaluable guidance throughout this research.

*"Rest at the End, Not in the Middle"*
