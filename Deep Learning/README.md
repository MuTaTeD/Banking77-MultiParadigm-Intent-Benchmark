# Track E: Pre-Transformer Deep Learning Baselines (Banking77)

This directory contains the complete experiment suite for **Track E (Pre-Transformer Deep Learning Baselines)** of the Banking77 Multi-Paradigm Intent Classification Benchmark.

---

## 1. Research Motivation & Literature Context

Before the dominance of Transformers (Devlin et al., 2019; Radford et al., 2019), conversational intent detection and sentence classification relied on word embeddings (GloVe, FastText) coupled with recurrent neural networks, temporal convolutions, and early attention pooling mechanisms. 

Evaluating these architectures on the exact same 77-class conversational banking distribution allows us to answer critical research questions:
1. **Receptive Field vs. Recurrence:** Do multi-scale convolutions capture local n-gram intent triggers better than sequential memory gates?
2. **Attention Pooling vs. Max/Mean Pooling:** How much gain does self-attention pooling (Bahdanau; Lin et al., 2017) provide over fixed 1-max pooling (Kim, 2014) or mean pooling?
3. **Representation Fusion:** Can combining convolutional feature maps with recurrent contextual states outperform either architecture in isolation?
4. **Computational Pareto Frontier:** What is the actual accuracy-per-second and accuracy-per-megabyte trade-off compared to fine-tuned BERT models and LoRA-adapted 3B/7B LLMs?

---

## 2. The 10 Benchmark Architectures

| # | Architecture Identifier | Type | Key Structural Characteristics | Primary Citation |
|---|-------------------------|------|--------------------------------|------------------|
| 1 | `FastText_Baseline` | Dense Averaging | GloVe 300d $\rightarrow$ Masked Global Average Pooling $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(77) | Joulin et al. (EMNLP 2017) |
| 2 | `TextCNN_Standard` | 1D-CNN | 3 parallel filter banks ($k \in \{2, 3, 4\}$, 100 filters each) $\rightarrow$ 1-Max Pooling $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(77) | Kim (EMNLP 2014) |
| 3 | `TextCNN_DeepMultiScale` | Deep 1D-CNN | 5 filter banks ($k \in \{2, 3, 4, 5, 6\}$, 128 filters each = 640d) + BatchNorm + 2-layer MLP head | Collobert et al. (JMLR 2011) |
| 4 | `BiLSTM_1Layer_MeanPool` | Recurrent (LSTM) | 1-layer BiLSTM ($h=256$, bidir=512) $\rightarrow$ Masked Global Mean Pooling $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(77) | Graves & Schmidhuber (2005) |
| 5 | `BiLSTM_2Layer_ConcatPool` | Stacked Recurrent | 2-layer stacked BiLSTM ($h=256$, dropout 0.3) $\rightarrow [\text{MeanPool}; \text{MaxPool}]$ (1024d) $\rightarrow$ Dense(256) $\rightarrow$ Dense(77) | Howard & Ruder (ACL 2018) |
| 6 | `BiLSTM_3Layer_Residual` | Deep Recurrent | 3-layer BiLSTM with skip connection ($L_3 + L_1$) $\rightarrow$ LayerNorm $\rightarrow$ Mean Pool $\rightarrow$ Dense(77) | He et al. (CVPR 2016) / Wu et al. |
| 7 | `BiLSTM_AdditiveAttention` | Attention Pooling | 1-layer BiLSTM ($h=256$) + Bahdanau-style Additive Self-Attention Context Pooling $\rightarrow$ Dense(77) | Lin et al. (ICLR 2017) |
| 8 | `BiGRU_MultiHeadAttention` | Multi-Head Attention | 2-layer BiGRU ($h=256$) + 4-Head Self-Attention Pooling + LayerNorm $\rightarrow$ Dense(77) | Zhou et al. (ACL 2016) |
| 9 | `Conv_BiLSTM_Sequential` | Sequential Hybrid | C-LSTM: Conv1D ($k=3$, 256 filters) $\rightarrow$ MaxPool($s=2$) $\rightarrow$ 1-layer BiLSTM $\rightarrow$ Mean Pool $\rightarrow$ Dense(77) | Zhou et al. (2015) |
| 10 | `DualStream_CNN_BiLSTM_Fusion` | Parallel Hybrid | Parallel Stream 1 (TextCNN 384d) $\oplus$ Stream 2 (BiLSTM-Attn 512d) $\rightarrow$ Concat (896d) $\rightarrow$ Dense(256) $\rightarrow$ Dense(77) | Zhang et al. (2018) |

---

## 3. Framework: PyTorch vs. Keras

### Can you use Keras or do you have to use PyTorch?
**You can use either! You do not have to use PyTorch.**
Both frameworks are fully supported in this folder:

1. **PyTorch Implementation (`models.py`, `train_deep_learning_suite.py`, and `Banking77_PreTransformer_DeepLearning_Suite.ipynb`):**
   - **Recommended for Benchmark Parity:** All other tracks in this project (Track A: SLMs/Unsloth, Track B: BERT family, Track C: Seq2Seq BART/T5) run on PyTorch.
   - **Hardware Telemetry:** PyTorch provides native, exact CUDA peak memory tracking (`torch.cuda.max_memory_allocated()`), allowing direct, uncompromised memory comparisons with BERT and LLM fine-tuning runs.
   - **Zero Configuration in Colab:** Works out-of-the-box on standard Colab GPU instances.

2. **Keras Implementation (`train_keras_suite.py`):**
   - **Keras 3 Multi-Backend Support:** Keras 3 can run directly on top of PyTorch (`export KERAS_BACKEND=torch`) or TensorFlow!
   - **Extremely Readable Functional API:** Writing multi-scale CNNs or dual-stream concatenations requires just ~15 lines using `layers.concatenate()`.

---

## 4. Benchmark Protocols & Setup

- **Dataset Split:** Official full Banking77 dataset (Casanueva et al., 2020) — **10,003 training queries, 0 validation split, 3,080 test queries**.
- **Random Seed:** Fixed to **`42`** across all data batching, initializations, and training loops.
- **Word Vectors:** Pretrained **GloVe 6B 300-dimensional** embeddings (Pennington et al., 2014) automatically downloaded and cached.
- **Optimization:** AdamW ($\text{lr}=10^{-3}$, $\text{weight decay}=10^{-4}$), Cosine Annealing learning rate schedule, Label Smoothing ($0.1$), and gradient clipping ($1.0$).
- **Telemetry Logged Automatically:**
  - Training time (seconds and formatted `Xm Ys`).
  - Inference / Test time (seconds and formatted).
  - Peak GPU VRAM allocated during training (`torch.cuda.max_memory_allocated()`).
  - Peak GPU VRAM allocated during testing.
  - Accuracy, Macro F1, Weighted F1, Macro/Weighted Precision and Recall.

---

## 5. How to Run

### Option A: Google Colab (Recommended)
1. Open `Banking77_PreTransformer_DeepLearning_Suite.ipynb` in Google Colab.
2. Ensure GPU runtime is selected (`Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ `T4 GPU`).
3. Run all cells. Checkpoints and consolidated CSV tables will automatically save to your Google Drive under:
   `MyDrive/NLP_Checkpoints/DeepLearning_training_Full10k_Awais_Seed42/`.
4. The notebook includes automatic resumption: if disconnected, re-running will load cached metrics and continue from the next model.

### Option B: Local CLI Execution
Run all 10 architectures sequentially:
```bash
python3 train_deep_learning_suite.py --model all --epochs 15 --batch_size 64 --seed 42 --output_dir ./results_deep_learning_seed42
```

Run a specific architecture (e.g. Dual-Stream CNN-BiLSTM Fusion):
```bash
python3 train_deep_learning_suite.py --model dualstream_cnn_bilstm_fusion --epochs 15
```

---

## 6. Output Artifacts

Each trained architecture creates:
- `model_best.pt`: Best checkpoint state dictionary.
- `metrics.json`: Full pre- and post-training classification metrics and hardware telemetry.
- `predictions.json`: Model predictions and ground-truth labels for all 3,080 test queries.
- `Track_E_Deep_Learning_Benchmark_Results.csv`: Master consolidated table comparing all 10 models side-by-side.
