"""
Track E: Pre-Transformer Deep Learning Baseline Architectures for Banking77
==========================================================================
10 Pre-Transformer architectures spanning:
1. FastText_Baseline          : Word Embedding Averaging (Joulin et al., 2016)
2. TextCNN_Standard           : Kim (2014) Multi-Scale 1D-CNN (kernels 2, 3, 4; 100 filters)
3. TextCNN_DeepMultiScale     : Extended 5-bank CNN (kernels 2, 3, 4, 5, 6; 128 filters) + BatchNorm + MLP
4. BiLSTM_1Layer_MeanPool     : 1-Layer Bidirectional LSTM (h=256) with Mean Pooling
5. BiLSTM_2Layer_ConcatPool   : 2-Layer Stacked BiLSTM with Concat Pooling [Mean; Max] (1024d)
6. BiLSTM_3Layer_Residual     : 3-Layer Deep BiLSTM with Residual Skip Connection (L3 + L1)
7. BiLSTM_AdditiveAttention   : 1-Layer BiLSTM with Bahdanau/Lin et al. (2017) Additive Attention
8. BiGRU_MultiHeadAttention   : 2-Layer BiGRU with 4-Head Self-Attention Pooling (Zhou et al., 2016)
9. Conv_BiLSTM_Sequential     : C-LSTM (Zhou et al., 2015) - Conv1D feature extractor -> BiLSTM
10. DualStream_CNN_BiLSTM_Fusion: Parallel CNN Stream + Attention-BiLSTM Stream Feature Fusion Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def create_embedding_layer(
    vocab_size: int,
    embedding_dim: int,
    pretrained_embeddings: Optional[torch.Tensor] = None,
    freeze_embeddings: bool = False
) -> nn.Embedding:
    """Helper to build embedding layer with optional pretrained weights."""
    if pretrained_embeddings is not None:
        emb = nn.Embedding.from_pretrained(
            pretrained_embeddings,
            freeze=freeze_embeddings,
            padding_idx=0
        )
    else:
        emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(emb.weight, mean=0.0, std=0.05)
        emb.weight.data[0].fill_(0.0)
    return emb


# =========================================================================
# Model 1: FastText Baseline (Joulin et al., EMNLP 2017)
# =========================================================================
class FastTextBaseline(nn.Module):
    """
    GloVe / FastText Embedding Averaging baseline.
    Computes masked average pooling over token embeddings -> Dropout -> Linear.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300, num_classes: int = 77,
                 pretrained_embeddings: Optional[torch.Tensor] = None, freeze_embeddings: bool = False,
                 dropout: float = 0.2):
        super().__init__()
        self.embedding = create_embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: [B, T]
        mask = (input_ids != 0).unsqueeze(-1)  # [B, T, 1]
        emb = self.embedding(input_ids)        # [B, T, D]
        emb = emb * mask
        sum_emb = emb.sum(dim=1)               # [B, D]
        lens = mask.sum(dim=1).clamp(min=1)    # [B, 1]
        pooled = sum_emb / lens
        out = self.dropout(pooled)
        return self.fc(out)


# =========================================================================
# Model 2: TextCNN Standard (Kim, EMNLP 2014)
# =========================================================================
class TextCNNStandard(nn.Module):
    """
    Classic Kim (2014) CNN with 3 parallel filter sizes (k=2, 3, 4), 100 filters each.
    1-Max Pooling -> Concat -> Dropout(0.5) -> Linear.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300, num_classes: int = 77,
                 pretrained_embeddings: Optional[torch.Tensor] = None, freeze_embeddings: bool = False,
                 filter_sizes: List[int] = [2, 3, 4], num_filters: int = 100, dropout: float = 0.5):
        super().__init__()
        self.embedding = create_embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, input_ids: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(input_ids).transpose(1, 2)  # [B, D, T]
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb))  # [B, num_filters, T - k + 1]
            p = F.adaptive_max_pool1d(c, 1).squeeze(-1)  # [B, num_filters]
            conv_outs.append(p)
        concat = torch.cat(conv_outs, dim=1)  # [B, len(filter_sizes) * num_filters]
        out = self.dropout(concat)
        return self.fc(out)


# =========================================================================
# Model 3: TextCNN Deep Multi-Scale Receptive Field
# =========================================================================
class TextCNNDeepMultiScale(nn.Module):
    """
    Extended 5-bank CNN (k=2, 3, 4, 5, 6; 128 filters each = 640d)
    with Batch Normalization, ReLU, Global Max Pooling, and a 2-layer MLP head.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300, num_classes: int = 77,
                 pretrained_embeddings: Optional[torch.Tensor] = None, freeze_embeddings: bool = False,
                 filter_sizes: List[int] = [2, 3, 4, 5, 6], num_filters: int = 128,
                 dense_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.embedding = create_embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in filter_sizes
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in filter_sizes])
        total_filters = len(filter_sizes) * num_filters
        self.fc1 = nn.Linear(total_filters, dense_dim)
        self.bn_dense = nn.BatchNorm1d(dense_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(input_ids).transpose(1, 2)  # [B, D, T]
        conv_outs = []
        for conv, bn in zip(self.convs, self.bns):
            c = F.relu(bn(conv(emb)))
            p = F.adaptive_max_pool1d(c, 1).squeeze(-1)
            conv_outs.append(p)
        concat = torch.cat(conv_outs, dim=1)
        h = F.relu(self.bn_dense(self.fc1(concat)))
        h = self.dropout(h)
        return self.fc2(h)


# =========================================================================
# Model 4: 1-Layer BiLSTM with Mean Pooling
# =========================================================================
class BiLSTM1LayerMeanPool(nn.Module):
    """
    Standard single-layer Bidirectional LSTM (hidden_size=256, bidir=512)
    with Global Mean Pooling over sequence time steps.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300, num_classes: int = 77,
                 pretrained_embeddings: Optional[torch.Tensor] = None, freeze_embeddings: bool = False,
                 hidden_size: int = 256, dropout: float = 0.3):
        super().__init__()
        self.embedding = create_embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(input_ids)  # [B, T, D]
        lstm_out, _ = self.lstm(emb)    # [B, T, 2H]
        # Masked mean pooling over non-pad tokens
        mask = (input_ids != 0).unsqueeze(-1)  # [B, T, 1]
        sum_out = (lstm_out * mask).sum(dim=1)
        lens = mask.sum(dim=1).clamp(min=1)
        mean_pooled = sum_out / lens
        out = self.dropout(mean_pooled)
        return self.fc(out)


# =========================================================================
# Model 5: 2-Layer Stacked BiLSTM with Concat Pooling [Mean; Max]
# =========================================================================
class BiLSTM2LayerConcatPool(nn.Module):
    """
    2-Layer Stacked BiLSTM with inter-layer dropout and Concat Pooling [MeanPool; MaxPool]
    (Howard & Ruder, ACL 2018 ULMFiT style pooling) yielding 1024d representation -> MLP.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300, num_classes: int = 77,
                 pretrained_embeddings: Optional[torch.Tensor] = None, freeze_embeddings: bool = False,
                 hidden_size: int = 256, dense_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.embedding = create_embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size * 2 * 2, dense_dim)  # [Mean; Max] = 4H
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)  # [B, T, 2H]
        mask = (input_ids != 0).unsqueeze(-1)

        # Masked Mean Pooling
        sum_out = (lstm_out * mask).sum(dim=1)
        lens = mask.sum(dim=1).clamp(min=1)
        mean_pool = sum_out / lens

        # Masked Max Pooling
        masked_out = lstm_out.masked_fill(~mask, -1e9)
        max_pool = masked_out.max(dim=1)[0]

        concat = torch.cat([mean_pool, max_pool], dim=1)  # [B, 4H]
        h = F.relu(self.fc1(concat))
        h = self.dropout(h)
        return self.fc2(h)


# =========================================================================
# Model 6: 3-Layer Deep BiLSTM with Residual Skip Connection
# =========================================================================
class BiLSTM3LayerResidual(nn.Module):
    """
    3-Layer Deep BiLSTM incorporating a highway/residual skip connection
    from Layer 1 output to Layer 3 output before classification head.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300, num_classes: int = 77,
                 pretrained_embeddings: Optional[torch.Tensor] = None, freeze_embeddings: bool = False,
                 hidden_size: int = 256, dropout: float = 0.4):
        super().__init__()
        self.embedding = create_embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size * 2, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(input_ids)
        out1, _ = self.lstm1(emb)       # [B, T, 2H]
        out2, _ = self.lstm2(out1)      # [B, T, 2H]
        out3, _ = self.lstm3(out2)      # [B, T, 2H]

        # Residual skip connection: Layer 3 + Layer 1
        res = self.layer_norm(out3 + out1)

        mask = (input_ids != 0).unsqueeze(-1)
        sum_out = (res * mask).sum(dim=1)
        lens = mask.sum(dim=1).clamp(min=1)
        pooled = sum_out / lens
        out = self.dropout(pooled)
        return self.fc(out)


# =========================================================================
# Model 7: BiLSTM with Additive Self-Attention Pooling (Lin et al., ICLR 2017)
# =========================================================================
class BiLSTMAdditiveAttention(nn.Module):
    """
    1-Layer BiLSTM with Bahdanau/Lin et al. (2017) Additive Attention pooling.
    a_t = softmax(v^T tanh(W h_t + b))
    c = sum_t a_t h_t
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300, num_classes: int = 77,
                 pretrained_embeddings: Optional[torch.Tensor] = None, freeze_embeddings: bool = False,
                 hidden_size: int = 256, attn_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        self.embedding = create_embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.w_attn = nn.Linear(hidden_size * 2, attn_dim)
        self.v_attn = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)  # [B, T, 2H]

        # Additive Attention Scoring
        u = torch.tanh(self.w_attn(lstm_out))  # [B, T, attn_dim]
        scores = self.v_attn(u).squeeze(-1)    # [B, T]

        # Mask padding tokens
        mask = (input_ids != 0)
        scores = scores.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]

        context = (lstm_out * attn_weights).sum(dim=1)  # [B, 2H]
        out = self.dropout(context)
        return self.fc(out)


# =========================================================================
# Model 8: 2-Layer BiGRU with Multi-Head Self-Attention Pooling
# =========================================================================
class BiGRUMultiHeadAttention(nn.Module):
    """
    2-Layer BiGRU with 4-Head Self-Attention Pooling (Zhou et al., 2016)
    allowing multiple focus points across conversational banking intents.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300, num_classes: int = 77,
                 pretrained_embeddings: Optional[torch.Tensor] = None, freeze_embeddings: bool = False,
                 hidden_size: int = 256, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.embedding = create_embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )
        self.num_heads = num_heads
        self.d_model = hidden_size * 2
        assert self.d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_model // num_heads

        # Multi-head attention query vectors
        self.head_queries = nn.Parameter(torch.randn(num_heads, self.d_model))
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(input_ids)
        gru_out, _ = self.gru(emb)  # [B, T, 2H]
        mask = (input_ids != 0)     # [B, T]

        # Multi-head attention pooling:
        # compute scores for each head: query [K, 2H] dot gru_out [B, T, 2H] -> [B, K, T]
        scores = torch.einsum("kh,bth->bkt", self.head_queries, gru_out) / math.sqrt(self.d_k)
        mask_expanded = mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(~mask_expanded, -1e9)
        attn = F.softmax(scores, dim=-1)  # [B, K, T]

        # Context per head: [B, K, T] @ [B, T, 2H] -> [B, K, 2H]
        heads_context = torch.bmm(attn, gru_out)  # [B, K, 2H]
        # Aggregate heads via mean
        multihead_context = heads_context.mean(dim=1)  # [B, 2H]
        normed = self.layer_norm(multihead_context)
        out = self.dropout(normed)
        return self.fc(out)


# =========================================================================
# Model 9: Sequential Conv-BiLSTM (C-LSTM, Zhou et al., 2015)
# =========================================================================
class ConvBiLSTMSequential(nn.Module):
    """
    C-LSTM (Zhou et al., 2015): 1D Convolution extracts local n-gram feature maps,
    subsampled via MaxPool1d, then passed sequentially into a BiLSTM.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300, num_classes: int = 77,
                 pretrained_embeddings: Optional[torch.Tensor] = None, freeze_embeddings: bool = False,
                 conv_filters: int = 256, kernel_size: int = 3, lstm_hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.embedding = create_embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)
        self.conv = nn.Conv1d(embedding_dim, conv_filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(conv_filters, lstm_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(input_ids).transpose(1, 2)  # [B, D, T]
        c = F.relu(self.conv(emb))                       # [B, conv_filters, T]
        p = self.pool(c).transpose(1, 2)                 # [B, T // 2, conv_filters]
        lstm_out, _ = self.lstm(p)                       # [B, T // 2, 2H]
        # Global mean pool across sequence
        mean_pool = lstm_out.mean(dim=1)
        out = self.dropout(mean_pool)
        return self.fc(out)


# =========================================================================
# Model 10: Dual-Stream CNN-BiLSTM Feature Fusion
# =========================================================================
class DualStreamCNNBiLSTMFusion(nn.Module):
    """
    Parallel Dual-Stream Architecture:
    - Stream 1: TextCNN with multi-scale kernels [2, 3, 4] -> MaxPool (384d)
    - Stream 2: BiLSTM + Additive Attention Context Vector (512d)
    - Fusion Head: Concat [Stream 1; Stream 2] (896d) -> Dense(256) -> Dense(77)
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300, num_classes: int = 77,
                 pretrained_embeddings: Optional[torch.Tensor] = None, freeze_embeddings: bool = False,
                 filter_sizes: List[int] = [2, 3, 4], num_filters: int = 128,
                 lstm_hidden: int = 256, dense_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.embedding = create_embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)

        # Stream 1: CNN
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        cnn_dim = len(filter_sizes) * num_filters

        # Stream 2: BiLSTM + Attention
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.w_attn = nn.Linear(lstm_hidden * 2, 128)
        self.v_attn = nn.Linear(128, 1, bias=False)
        lstm_dim = lstm_hidden * 2

        # Fusion Head
        total_dim = cnn_dim + lstm_dim
        self.fc1 = nn.Linear(total_dim, dense_dim)
        self.bn_dense = nn.BatchNorm1d(dense_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(input_ids)  # [B, T, D]

        # --- Stream 1: CNN ---
        emb_trans = emb.transpose(1, 2)
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb_trans))
            p = F.adaptive_max_pool1d(c, 1).squeeze(-1)
            conv_outs.append(p)
        cnn_feat = torch.cat(conv_outs, dim=1)  # [B, cnn_dim]

        # --- Stream 2: BiLSTM + Attention ---
        lstm_out, _ = self.lstm(emb)
        u = torch.tanh(self.w_attn(lstm_out))
        scores = self.v_attn(u).squeeze(-1)
        mask = (input_ids != 0)
        scores = scores.masked_fill(~mask, -1e9)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)
        lstm_feat = (lstm_out * attn).sum(dim=1)  # [B, lstm_dim]

        # --- Fusion ---
        fused = torch.cat([cnn_feat, lstm_feat], dim=1)
        h = F.relu(self.bn_dense(self.fc1(fused)))
        h = self.dropout(h)
        return self.fc2(h)


# =========================================================================
# Model Registry & Factory
# =========================================================================
MODEL_REGISTRY = {
    "fasttext_baseline": FastTextBaseline,
    "textcnn_standard": TextCNNStandard,
    "textcnn_deep_multiscale": TextCNNDeepMultiScale,
    "bilstm_1layer_meanpool": BiLSTM1LayerMeanPool,
    "bilstm_2layer_concatpool": BiLSTM2LayerConcatPool,
    "bilstm_3layer_residual": BiLSTM3LayerResidual,
    "bilstm_additive_attention": BiLSTMAdditiveAttention,
    "bigru_multihead_attention": BiGRUMultiHeadAttention,
    "conv_bilstm_sequential": ConvBiLSTMSequential,
    "dualstream_cnn_bilstm_fusion": DualStreamCNNBiLSTMFusion,
}


def get_model(
    model_name: str,
    vocab_size: int,
    embedding_dim: int = 300,
    num_classes: int = 77,
    pretrained_embeddings: Optional[torch.Tensor] = None,
    freeze_embeddings: bool = False
) -> nn.Module:
    """Factory function to instantiate any of the 10 models by slug."""
    model_name_clean = model_name.lower().replace("-", "_")
    if model_name_clean not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_name_clean]
    return model_class(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=freeze_embeddings
    )
