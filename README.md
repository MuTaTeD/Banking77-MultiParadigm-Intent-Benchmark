# Natural Language Processing (CSAI8333) — Course Assignments

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Unsloth%20%7C%20HuggingFace%20%7C%20PyTorch-orange.svg)](https://github.com/unslothai/unsloth)
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)]()

This repository contains academic coursework, experimental implementations, and executed Jupyter notebooks for **Special Topics in Natural Language Processing (CSAI8333)**, PhD in Computer Science, Fall 2026 at the **University of Central Punjab (UCP)**, Lahore, Pakistan.

* **Public Repository:** [https://github.com/MuTaTeD/NLP-Assignments](https://github.com/MuTaTeD/NLP-Assignments)

---

## 👥 Project Team & Authors

* **Awais M. Lodhi** (ID: `L1F26PHDC0006`)
* **Sajid Saleem** (ID: `L1F26PHDC0001`)
* **Basit Ali Gillani** (ID: `L1F26PHDC0002`)
* **Dr. Ali Saeed** *(Faculty / Course Instructor)*

*Program:* PhD in Computer Science (Fall 2026)  
*Department:* Department of Computer Science, Faculty of Information Technology  
*Institution:* University of Central Punjab (UCP), Lahore, Pakistan  

---

## 📂 Repository Structure

```text
NLP-Assignments/
├── README.md
├── .gitignore
├── Bert Finetuning/                             # Track B: 12 BERT-Family Encoder Architectures
│   ├── Banking77_BERT_MultiModel_Tuning - (Awais Lodhi).ipynb   # Seed: 42
│   ├── Banking77_BERT_MultiModel_Tuning - (Sajid Saleem).ipynb  # Seed: 1337
│   ├── Banking77_BERT_MultiModel_Tuning - (Basit Gillani).ipynb # Seed: 2026
│   └── Banking77_BERT_MultiModel_Tuning.ipynb                   # Master Template
└── LLM Finetuning/                              # Track A: 6 Instruction-Tuned LLMs/SLMs
    ├── Gemma_2_2B_IT_Banking_Classification - (Awais Lodhi).ipynb   # Validation Run
    ├── Gemma_2_2B_IT_Banking_Classification - (Sajid Saleem).ipynb  # Evaluation Run
    ├── Llama_3_2_3B_Banking_Classification - (Awais Lodhi).ipynb
    ├── Mistral_7B_Banking_Classification - (Awais Lodhi).ipynb
    ├── Phi_3_5_mini_Banking_Classification - (Basit Gillani).ipynb
    ├── Qwen2_5_3B_Banking_Classification - (Basit Gillani).ipynb
    └── TinyLlama_1_1B_Banking_Classification - (Sajid Saleem).ipynb
```

---

## 🎯 Benchmark Scope & Unified Dataset Splits

All models in both Track A (LLMs) and Track B (BERTs) are benchmarked on the standardized **Banking77** (`PolyAI/banking77`) fine-grained intent classification benchmark across **77 canonical customer service classes**:

| Split | Instances | Proportion | Description |
| :--- | :---: | :---: | :--- |
| **Training Set** (`train.csv`) | **9,003** | 68.8% | Supervised training split |
| **Validation Set** (`validation.csv`) | **1,000** | 7.6% | Deterministic split from train (`seed=42`) for hyperparameter evaluation & checkpointing |
| **Test Set** (`test.csv`) | **3,080** | 23.5% | Official untouched benchmark test set evaluated identically across all architectures |
| **Total Benchmark Volume** | **13,083** | 100.0% | Complete Banking77 customer query corpus across 77 intent classes |

---

## 🚀 Assignment #1: Comparative Intent Classification Across Modern NLP Paradigms

This assignment investigates two dominant natural language processing paradigms for high-cardinality multi-class classification:

### Track A: Parameter-Efficient Fine-Tuning (4-bit QLoRA) on 6 LLMs/SLMs
Evaluates modern open-weight Instruction-tuned Small and Large Language Models (1.1B to 7.2B parameters) using **Unsloth** accelerated QLoRA ($r=16, \alpha=16$):
1. **TinyLlama 1.1B Chat**
2. **Gemma 2 2B IT**
3. **Llama 3.2 3B Instruct**
4. **Qwen2.5-3B-Instruct**
5. **Phi-3.5-mini-Instruct (3.8B)**
6. **Mistral 7B Instruct v0.3**

Evaluated across three prompting strategies pre- and post-adaptation:
- **Baseline (Zero-Shot):** Direct taxonomy-constrained intent extraction.
- **Few-Shot:** In-context demonstration exemplars.
- **Feature-Aware:** Structured transactional entity and action state extraction.

### Track B: Multi-Model Full Fine-Tuning on 12 BERT-Family Architectures
Evaluates representation-learning encoder models spanning general, domain-specific (finance, biomedicine, science), and compressed architectures across 3 independent team seeds (`42`, `1337`, `2026`) for statistical variance and reproducibility analysis:
1. **BERT-Base** (`bert-base-uncased` - 110M)
2. **DistilBERT** (`distilbert-base-uncased` - 66M)
3. **SciBERT** (`allenai/scibert_scivocab_uncased` - 110M)
4. **BioBERT** (`dmis-lab/biobert-v1.1` - 110M)
5. **PubMedBERT** (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` - 110M)
6. **BERT-Tiny** (`prajjwal1/bert-tiny` - 4.4M)
7. **mBERT** (`bert-base-multilingual-cased` - 178M)
8. **BERT-Large** (`bert-large-uncased` - 340M)
9. **RoBERTa** (`roberta-base` - 125M)
10. **ALBERT** (`albert-base-v2` - 12M)
11. **FinBERT** (`ProsusAI/finbert` - 110M)
12. **DeBERTa-v3** (`microsoft/deberta-v3-base` - 86M)

---

## ⚡ Empirical Hardware & Resource Profiles (Nvidia Tesla T4)

* **BERT Full Training (All 12 Models):** ~1.5 to 2.0 hours total runtime on a single T4 GPU.
* **LLM QLoRA Fine-Tuning:** 1.5 to 3.0 hours per model on 9,003 queries (2 epochs).
* **VRAM Footprint:**
  * **BERT-Tiny:** < 1.0 GB VRAM
  * **BERT-Base / RoBERTa / DeBERTa-v3:** ~2.5 - 3.5 GB VRAM
  * **BERT-Large:** ~6.5 GB VRAM
  * **TinyLlama 1.1B (4-bit QLoRA):** ~3.5 GB VRAM
  * **SLMs (Gemma 2, Llama 3.2, Qwen2.5, Phi-3.5):** ~6.0 - 7.0 GB VRAM
  * **Mistral 7B Instruct:** ~13.0 GB VRAM

---

## 🔧 Environment Setup & Execution

Each notebook is self-contained and ready for execution on **Google Colab (T4 GPU)** or **Kaggle**:

```bash
# Core Environment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft bitsandbytes scikit-learn pandas tqdm evaluate gradio
# For LLM QLoRA Acceleration
pip install unsloth "xformers<0.0.28" "trl<0.9.0"
```

1. Open any notebook in Google Colab or Kaggle.
2. Select **GPU runtime** (Nvidia T4 or P100).
3. Run all cells sequentially. Results, loss logs, and checkpoints are automatically saved.
