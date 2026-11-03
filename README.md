# Banking77 Multi-Paradigm Intent Classification Benchmark (4-Bit QLoRA LLMs/SLMs vs. Multi-Seed BERT Encoders)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Unsloth%20%7C%20HuggingFace%20%7C%20PyTorch-orange.svg)](https://github.com/unslothai/unsloth)
[![Dataset](https://img.shields.io/badge/Dataset-PolyAI%2FBanking77-green.svg)](https://huggingface.co/datasets/PolyAI/banking77)
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)]()

This repository contains the executed Jupyter notebooks, multi-paradigm empirical evaluation results, and hardware/runtime telemetry for **Special Topics in Natural Language Processing (CSAI8333)**, PhD in Computer Science (Fall 2026) at the **University of Central Punjab (UCP)**, Lahore, Pakistan.

* **Public Repository:** [https://github.com/MuTaTeD/Banking77-MultiParadigm-Intent-Benchmark](https://github.com/MuTaTeD/Banking77-MultiParadigm-Intent-Benchmark)

---

## 👥 Project Team & Authors

| Author Name | Registration / Role | Email Address | ORCID |
| :--- | :--- | :--- | :---: |
| **Awais Muhammad Lodhi** | `L1F26PHDC0006` (PhD CS Scholar) | `awais.lodhi@ucp.edu.pk` | — |
| **Sajid Saleem** | `L1F26PHDC0001` (PhD CS Scholar) | `sajid.saleem@ucp.edu.pk` | — |
| **Basit Ali Gillani** | `L1F26PHDC0002` (PhD CS Scholar) | `basit.gilani@ucp.edu.pk` | — |
| **Dr. Ali Saeed** | Faculty / Course Instructor & Supervisor | `ali.saeed@ucp.edu.pk` | — |

*Program:* PhD in Computer Science (Fall 2026)  
*Department:* Department of Computer Science, Faculty of Information Technology  
*Institution:* University of Central Punjab (UCP), Lahore, Pakistan  

---

## 🏗 Project Pipeline Architecture

![End-to-End Multi-Paradigm Project Pipeline Architecture](figures/pipeline_architecture.png)

---

## 📂 Repository Structure

```text
Banking77-MultiParadigm-Intent-Benchmark/
├── README.md
├── .gitignore
├── figures/                                     # 300-DPI Publication Figures
│   ├── pipeline_architecture.png
│   ├── pre_vs_post_leap.png
│   ├── post_tuning_accuracy_comparison.png
│   └── parameter_vs_accuracy_frontier.png
├── Bert Finetuning/                             # Track B: 12 BERT-Family Encoder Architectures
│   ├── Banking77_BERT_MultiModel_Tuning_Awais_Lodhi.ipynb       # Seed: 42 (Awais M. Lodhi)
│   ├── Banking77_BERT_MultiModel_Tuning_Sajid_Saleem.ipynb      # Seed: 1337 (Sajid Saleem)
│   └── Banking77_BERT_MultiModel_Tuning_Basit_Gillani.ipynb     # Seed: 2026 (Basit A. Gillani)
└── LLM Finetuning/                              # Track A: 6 Instruction-Tuned LLMs/SLMs (7 Runs)
    ├── Gemma_2_2B_IT_Banking_Classification_Awais_Lodhi.ipynb   # Verification Run (Awais M. Lodhi)
    ├── Gemma_2_2B_IT_Banking_Classification_Sajid_Saleem.ipynb  # Primary Run (Sajid Saleem)
    ├── Llama_3_2_3B_Banking_Classification_Awais_Lodhi.ipynb    # Awais M. Lodhi
    ├── Mistral_7B_Banking_Classification_Awais_Lodhi.ipynb      # Awais M. Lodhi
    ├── Phi_3_5_mini_Banking_Classification_Basit_Gillani.ipynb  # Basit A. Gillani
    ├── Qwen2_5_3B_Banking_Classification_Basit_Gillani.ipynb    # Basit A. Gillani
    └── TinyLlama_1_1B_Banking_Classification_Sajid_Saleem.ipynb # Sajid Saleem
```

---

## 🎯 Benchmark Scope & Zero-Leakage Dataset Splits

All models in both **Track A (Generative LLMs/SLMs)** and **Track B (Discriminative BERT Encoders)** are evaluated on the standardized **Banking77** (`PolyAI/banking77`) fine-grained intent classification benchmark across **77 canonical customer service classes** under a strict zero-leakage split:

| Split | Instances | Proportion | Description |
| :--- | :---: | :---: | :--- |
| **Training Set** (`train.csv`) | **9,003** | 68.81% | Supervised training split ($\approx 117$ queries/class) |
| **Validation Set** (`validation.csv`) | **1,000** | 7.64% | Deterministic split from raw training pool (`seed=42`) for validation & checkpointing |
| **Test Set** (`test.csv`) | **3,080** | 23.54% | Official untouched benchmark test set (40 queries $\times$ 77 classes) evaluated identically across all models |
| **Total Benchmark Volume** | **13,083** | 100.00% | Complete Banking77 customer query corpus across 77 intent classes |

---

## ⚡ Experimental Setup, Model Parameters, Runtime & GPU Telemetry

All experiments were executed on **Google Colab** using single **NVIDIA Tesla T4 GPUs** (`14.56 GB` / `15.64 GB` VRAM). To keep the main classification benchmark table focused strictly on classification metrics, all parameter counts, execution times, and GPU VRAM telemetry are reported here.

### Track A: Generative LLMs / SLMs (4-Bit NF4 QLoRA via Unsloth, 2 Epochs, `r=16, α=16`)

| # | Model Architecture | Params | QLoRA Train Time | Train VRAM | Peak VRAM | Total Runtime | Platform / GPU |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :--- |
| 1 | **TinyLlama 1.1B Chat** | 1.1B | `00:28:56` (28.9m) | 1.7 GB | 5.3 GB | `00:46:01` (46.0m) | Colab / Tesla T4 (14.56 GB) |
| 2 | **Gemma 2 2B IT (Primary Run)** | 2.6B | `01:05:15` (65.3m) | 3.4 GB | 9.6 GB | `01:28:13` (88.2m) | Colab / Tesla T4 (14.56 GB) |
| 3 | **Gemma 2 2B IT (Verification Run)** | 2.6B | `01:06:37` (66.6m) | 3.4 GB | 9.6 GB | `01:29:56` (89.9m) | Colab / Tesla T4 (14.56 GB) |
| 4 | **Llama 3.2 3B Instruct** | 3.2B | `01:00:01` (60.0m) | 3.7 GB | 11.7 GB | `01:27:41` (87.7m) | Colab / Tesla T4 (14.56 GB) |
| 5 | **Qwen2.5-3B-Instruct** | 3.1B | `00:59:46` (59.8m) | 5.3 GB | 7.7 GB | `02:00:28` (120.5m) | Colab / Tesla T4 (14.56 GB) |
| 6 | **Phi-3.5-mini-Instruct** | 3.8B | `01:44:20` (104.3m) | 4.9 GB | 14.5 GB | `02:32:05` (152.1m) | Colab / Tesla T4 (14.56 GB) |
| 7 | **Mistral 7B Instruct v0.3** | 7.2B | `02:27:43` (147.7m) | 6.0 GB | 10.6 GB | `03:38:42` (218.7m) | Colab / Tesla T4 (14.56 GB) |

### Track B: Discriminative BERT-Family Encoders (Full Fine-Tuning, 3 Epochs, Seeds 42, 1337, 2026)

| # | Model Name | HuggingFace Checkpoint ID | Params | Batch / Accum / LR | Mean Train Time ($\mu \pm \sigma$) | Peak VRAM (Alloc / Res) |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: |
| 1 | **BERT-Base** | `bert-base-uncased` | 110M | 32 / 1 / `3e-5` | $5.10 \pm 0.08\text{ min}$ | 2.47 / 2.81 GB |
| 2 | **DistilBERT** | `distilbert-base-uncased` | 66M | 32 / 1 / `3e-5` | $2.65 \pm 0.05\text{ min}$ | 1.40 / 1.70 GB |
| 3 | **SciBERT** | `allenai/scibert_scivocab_uncased` | 110M | 32 / 1 / `3e-5` | $5.17 \pm 0.09\text{ min}$ | 2.48 / 2.84 GB |
| 4 | **BioBERT v1.1** | `dmis-lab/biobert-v1.1` | 110M | 32 / 1 / `3e-5` | $5.18 \pm 0.10\text{ min}$ | 2.47 / 2.71 GB |
| 5 | **PubMedBERT** | `microsoft/BiomedNLP-PubMedBERT...` | 110M | 32 / 1 / `3e-5` | $5.18 \pm 0.10\text{ min}$ | 2.48 / 2.83 GB |
| 6 | **BERT-Tiny** | `prajjwal1/bert-tiny` | 4.4M | 64 / 1 / `5e-5` | $0.14 \pm 0.00\text{ min}$ | 0.52 / 0.55 GB |
| 7 | **mBERT** | `bert-base-multilingual-cased` | 178M | 32 / 1 / `3e-5` | $5.57 \pm 0.09\text{ min}$ | 3.34 / 4.20 GB |
| 8 | **BERT-Large** | `bert-large-uncased` | 340M | 16 / 2 / `2e-5` | $18.15 \pm 0.35\text{ min}$ | 6.62 / 6.65 GB |
| 9 | **RoBERTa-Base** | `roberta-base` | 125M | 32 / 1 / `2e-5` | $5.24 \pm 0.12\text{ min}$ | 2.93 / 3.58 GB |
| 10 | **ALBERT-Base-v2** | `albert-base-v2` | 12M | 32 / 1 / `3e-5` | $5.27 \pm 0.10\text{ min}$ | 2.52 / 2.64 GB |
| 11 | **FinBERT** | `ProsusAI/finbert` | 110M | 32 / 1 / `3e-5` | $5.18 \pm 0.11\text{ min}$ | 2.48 / 2.81 GB |
| 12 | **DeBERTa-v3-Base** | `microsoft/deberta-v3-base` | 86M | 16 / 2 / `2e-5` | $2.79 \pm 0.09\text{ min}$ | 2.42 / 2.50 GB |
| — | **Total Suite** | *Complete 12-Model Execution* | — | — | $\mathbf{65.62 \pm 1.26\text{ min}}$ | **6.62 / 6.65 GB** |

---

## 📊 Single Consolidated Master Benchmark Results Table (`N = 3,080` Test Queries)

| Model Architecture | Prompt / Seed Strategy | Pre Acc (%) | Post Acc (%) | Post Prec (%) | Post Rec (%) | Post $F_1$ (%) | Acc Gain ($\Delta$%) | $F_1$ Gain ($\Delta$%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Section A: Track A — Generative LLMs / SLMs (4-Bit QLoRA Fine-Tuning across 3 Prompting Regimes)** | | | | | | | | |
| **TinyLlama 1.1B Chat** | Baseline (Zero-Shot) | 0.00 | 90.81 | 91.12 | 90.81 | 90.85 | +90.81 | +90.85 |
| | Few-Shot Exemplars | 0.00 | 90.52 | 91.16 | 90.52 | 90.69 | +90.52 | +90.69 |
| | Feature-Aware | 0.00 | 90.58 | 91.00 | 90.58 | 90.67 | +90.58 | +90.67 |
| **Gemma 2 2B IT (Primary Run)** | Baseline (Zero-Shot) | 0.10 | **93.83** | **94.11** | **93.83** | **93.87** | **+93.73** | **+93.69** |
| | Few-Shot Exemplars | 6.10 | 93.67 | 93.92 | 93.67 | 93.70 | +87.57 | +87.12 |
| | Feature-Aware | 0.03 | 93.54 | 93.74 | 93.54 | 93.55 | +93.51 | +93.49 |
| **Gemma 2 2B IT (Verification Run)** | Baseline (Zero-Shot) | 0.10 | 93.60 | 93.88 | 93.60 | 93.64 | +93.50 | +93.46 |
| | Few-Shot Exemplars | 6.07 | **93.86** | **94.11** | **93.86** | **93.89** | +87.79 | +87.36 |
| | Feature-Aware | 0.03 | 93.47 | 93.76 | 93.47 | 93.51 | +93.44 | +93.45 |
| **Llama 3.2 3B Instruct** | Baseline (Zero-Shot) | 0.00 | 92.53 | 93.05 | 92.53 | 92.63 | +92.53 | +92.63 |
| | Few-Shot Exemplars | 5.13 | 92.99 | 93.42 | 92.99 | 93.08 | +87.86 | +89.58 |
| | Feature-Aware | 0.19 | 92.60 | 93.17 | 92.60 | 92.62 | +92.41 | +92.26 |
| **Qwen2.5-3B-Instruct** | Baseline (Zero-Shot) | 0.00 | 92.86 | 93.14 | 92.86 | 92.88 | +92.86 | +92.88 |
| | Few-Shot Exemplars | **6.98** | 92.73 | 93.01 | 92.73 | 92.76 | +85.75 | +84.88 |
| | Feature-Aware | 0.03 | 92.73 | 93.20 | 92.73 | 92.82 | +92.70 | +92.76 |
| **Phi-3.5-mini-Instruct** | Baseline (Zero-Shot) | 0.00 | 91.56 | 91.77 | 91.56 | 91.53 | +91.56 | +91.53 |
| | Few-Shot Exemplars | 5.00 | 91.82 | 91.90 | 91.82 | 91.77 | +86.82 | +87.04 |
| | Feature-Aware | 0.00 | 91.95 | 92.07 | 91.95 | 91.89 | +91.95 | +91.89 |
| **Mistral 7B Instruct v0.3** | Baseline (Zero-Shot) | 0.19 | 93.02 | 93.25 | 93.02 | 93.01 | +92.83 | +92.67 |
| | Few-Shot Exemplars | 3.77 | 93.21 | 93.38 | 93.21 | 93.19 | +89.44 | +88.89 |
| | Feature-Aware | 0.06 | 93.02 | 93.16 | 93.02 | 92.98 | +92.96 | +92.86 |
| **Section B: Track B — Discriminative BERT Encoders (Multi-Seed Aggregation: Mean ± Variance across Seeds 42, 1337, 2026)** | | | | | | | | |
| **BERT-Base** | Multi-Seed (Mean ± Var) | 1.34 ± 0.02 | 85.08 ± 0.91 | 85.15 ± 0.31 | 85.08 ± 0.91 | 83.77 ± 1.20 | +83.73 ± 1.23 | +83.68 ± 1.23 |
| **DistilBERT** | Multi-Seed (Mean ± Var) | 1.06 ± 0.08 | 83.79 ± 0.17 | 83.79 ± 0.18 | 83.79 ± 0.17 | 82.62 ± 0.15 | +82.73 ± 0.50 | +82.48 ± 0.27 |
| **SciBERT** | Multi-Seed (Mean ± Var) | 1.01 ± 0.18 | **89.62 ± 0.01** | **90.33 ± 0.02** | **89.62 ± 0.01** | **89.60 ± 0.00** | **+88.62 ± 0.26** | **+89.37 ± 0.03** |
| **BioBERT v1.1** | Multi-Seed (Mean ± Var) | 1.19 ± 0.23 | 85.09 ± 0.49 | 85.67 ± 0.33 | 85.09 ± 0.49 | 84.16 ± 0.54 | +83.89 ± 0.54 | +84.06 ± 0.50 |
| **PubMedBERT** | Multi-Seed (Mean ± Var) | 1.31 ± 0.01 | 87.71 ± 0.03 | 88.55 ± 0.59 | 87.71 ± 0.03 | 87.44 ± 0.16 | +86.41 ± 0.06 | +87.23 ± 0.15 |
| **BERT-Tiny** | Multi-Seed (Mean ± Var) | 1.42 ± 0.16 | 20.02 ± 0.63 | 22.78 ± 1.40 | 20.02 ± 0.63 | 15.40 ± 0.10 | +18.60 ± 1.15 | +15.23 ± 0.16 |
| **mBERT** | Multi-Seed (Mean ± Var) | 1.21 ± 0.02 | 84.27 ± 0.35 | 84.79 ± 1.58 | 84.27 ± 0.35 | 83.12 ± 0.53 | +83.06 ± 0.39 | +83.01 ± 0.50 |
| **BERT-Large** | Multi-Seed (Mean ± Var) | 1.34 ± 0.00 | 86.84 ± 2.60 | 87.43 ± 3.37 | 86.84 ± 2.60 | 86.06 ± 4.44 | +85.50 ± 2.55 | +85.95 ± 4.38 |
| **RoBERTa-Base** | Multi-Seed (Mean ± Var) | 1.30 ± 0.00 | 88.60 ± 0.09 | 89.28 ± 1.02 | 88.60 ± 0.09 | 88.22 ± 0.26 | +87.31 ± 0.09 | +88.19 ± 0.26 |
| **ALBERT-Base-v2** | Multi-Seed (Mean ± Var) | 1.24 ± 0.00 | 77.86 ± 0.23 | 79.12 ± 1.22 | 77.86 ± 0.23 | 76.31 ± 0.11 | +76.61 ± 0.27 | +76.11 ± 0.12 |
| **FinBERT** | Multi-Seed (Mean ± Var) | 1.18 ± 0.33 | 74.72 ± 0.31 | 75.07 ± 1.35 | 74.72 ± 0.31 | 71.56 ± 0.56 | +73.54 ± 0.12 | +71.36 ± 0.42 |
| **DeBERTa-v3-Base** | Multi-Seed (Mean ± Var) | 1.31 ± 0.00 | 1.30 ± 0.00 | 0.02 ± 0.00 | 1.30 ± 0.00 | 0.03 ± 0.00 | -0.01 ± 0.00 | -0.01 ± 0.00 |

## 📂 Data Availability Statement

The canonical **Banking77** dataset (`PolyAI/banking77`, 13,083 customer service queries across 77 intent categories) is publicly available on the HuggingFace Datasets Hub at [https://huggingface.co/datasets/PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77). All 10 executed Jupyter notebooks (7 Track A LLM/SLM notebooks and 3 Track B 12-model BERT notebooks) containing complete cell outputs, stage-wise execution logs, and evaluation telemetry are openly available in this repository.

---

## 🏅 CRediT Author Contribution Statement & Credits

- **Awais Muhammad Lodhi** (`L1F26PHDC0006`, `awais.lodhi@ucp.edu.pk`): *Conceptualization, Methodology, Software, Data Curation, Investigation, Formal Analysis, Visualization, Writing – Original Draft & Editing.* Designed the multi-paradigm benchmarking architecture and zero-leakage dataset stratification protocol; executed Track A QLoRA fine-tuning and multi-prompt evaluations for **Llama 3.2 3B Instruct**, **Mistral 7B Instruct v0.3**, and the independent verification run of **Gemma 2 2B IT**; executed the primary Track B multi-model benchmark across all 12 BERT architectures (**Seed 42**).
- **Sajid Saleem** (`L1F26PHDC0001`, `sajid.saleem@ucp.edu.pk`): *Methodology, Investigation, Validation, Resources, Data Curation, Writing – Review & Editing.* Executed Track A QLoRA fine-tuning and 3-strategy prompt evaluations for **TinyLlama 1.1B Chat** and the primary benchmark run of **Gemma 2 2B IT**; executed the first independent Track B replication sweep across all 12 BERT architectures (**Seed 1337**).
- **Basit Ali Gillani** (`L1F26PHDC0002`, `basit.gilani@ucp.edu.pk`): *Methodology, Investigation, Validation, Resources, Data Curation, Writing – Review & Editing.* Executed Track A QLoRA fine-tuning and 3-strategy prompt evaluations for **Qwen2.5-3B-Instruct** and **Phi-3.5-mini-Instruct**; executed the second independent Track B replication sweep across all 12 BERT architectures (**Seed 2026**).
- **Dr. Ali Saeed** (`ali.saeed@ucp.edu.pk`): *Supervision, Project Administration, Conceptualization, Methodology, Validation, Writing – Review & Editing.* Formulated the research scope and experimental rigor standards for Special Topics in Natural Language Processing (CSAI8333) at the University of Central Punjab (UCP) and supervised the study.
