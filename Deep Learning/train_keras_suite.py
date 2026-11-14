"""
Track E: Keras 3 / TensorFlow Implementation Suite for Banking77
================================================================
Demonstrates the equivalent implementations of the 10 Pre-Transformer architectures
using Keras Functional API.

Can run with:
1. TensorFlow Backend: default `tf.keras` or `os.environ["KERAS_BACKEND"] = "tensorflow"`
2. PyTorch Backend: Keras 3 with `os.environ["KERAS_BACKEND"] = "torch"`
"""

import os
import time
import json
import argparse
from typing import Dict, Any, List

# Select backend if using Keras 3 (e.g. "tensorflow" or "torch")
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "tensorflow"

try:
    import keras
    from keras import layers, Model
except ImportError:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def build_keras_model(model_name: str, vocab_size: int, embedding_dim: int = 300, num_classes: int = 77, max_len: int = 64) -> Model:
    """
    Factory function for all 10 Pre-Transformer architectures using Keras Functional API.
    """
    inputs = layers.Input(shape=(max_len,), dtype="int32", name="input_ids")
    # Mask zero for padding tokens
    emb = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=False, name="embedding")(inputs)

    name = model_name.lower()

    # 1. FastText Baseline
    if name == "fasttext_baseline":
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(emb)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    # 2. TextCNN Standard (Kim, 2014)
    elif name == "textcnn_standard":
        conv_pools = []
        for k in [2, 3, 4]:
            c = layers.Conv1D(filters=100, kernel_size=k, activation="relu")(emb)
            p = layers.GlobalMaxPooling1D()(c)
            conv_pools.append(p)
        x = layers.concatenate(conv_pools, axis=-1)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    # 3. TextCNN Deep Multi-Scale
    elif name == "textcnn_deep_multiscale":
        conv_pools = []
        for k in [2, 3, 4, 5, 6]:
            c = layers.Conv1D(filters=128, kernel_size=k, padding="same")(emb)
            c = layers.BatchNormalization()(c)
            c = layers.Activation("relu")(c)
            p = layers.GlobalMaxPooling1D()(c)
            conv_pools.append(p)
        x = layers.concatenate(conv_pools, axis=-1)
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    # 4. BiLSTM 1-Layer MeanPool
    elif name == "bilstm_1layer_meanpool":
        lstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(emb)
        x = layers.GlobalAveragePooling1D()(lstm)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    # 5. BiLSTM 2-Layer ConcatPool
    elif name == "bilstm_2layer_concatpool":
        lstm1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(emb)
        lstm2 = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(lstm1)
        avg_p = layers.GlobalAveragePooling1D()(lstm2)
        max_p = layers.GlobalMaxPooling1D()(lstm2)
        x = layers.concatenate([avg_p, max_p], axis=-1)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    # 6. BiLSTM 3-Layer Residual
    elif name == "bilstm_3layer_residual":
        l1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(emb)
        l2 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(l1)
        l3 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(l2)
        # Residual add: L3 + L1
        res = layers.add([l3, l1])
        res = layers.LayerNormalization()(res)
        x = layers.GlobalAveragePooling1D()(res)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    # 7. BiLSTM Additive Attention
    elif name == "bilstm_additive_attention":
        lstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(emb)
        # Self-Attention scoring
        u = layers.Dense(128, activation="tanh")(lstm)
        scores = layers.Dense(1, use_bias=False)(u)
        weights = layers.Softmax(axis=1)(scores)
        ctx = layers.Dot(axes=1)([weights, lstm])
        ctx = layers.Reshape((512,))(ctx)
        x = layers.Dropout(0.4)(ctx)
        outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    # 8. BiGRU Multi-Head Attention
    elif name == "bigru_multihead_attention":
        gru = layers.Bidirectional(layers.GRU(256, return_sequences=True, dropout=0.3))(emb)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(gru, gru)
        x = layers.LayerNormalization()(attn)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    # 9. Conv-BiLSTM Sequential (C-LSTM)
    elif name == "conv_bilstm_sequential":
        c = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(emb)
        p = layers.MaxPooling1D(pool_size=2)(c)
        lstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(p)
        x = layers.GlobalAveragePooling1D()(lstm)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    # 10. Dual-Stream CNN-BiLSTM Feature Fusion
    elif name == "dualstream_cnn_bilstm_fusion":
        # Stream 1: CNN
        conv_pools = []
        for k in [2, 3, 4]:
            c = layers.Conv1D(128, kernel_size=k, activation="relu")(emb)
            p = layers.GlobalMaxPooling1D()(c)
            conv_pools.append(p)
        cnn_feat = layers.concatenate(conv_pools, axis=-1)

        # Stream 2: BiLSTM + Attention
        lstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(emb)
        u = layers.Dense(128, activation="tanh")(lstm)
        scores = layers.Dense(1, use_bias=False)(u)
        weights = layers.Softmax(axis=1)(scores)
        lstm_ctx = layers.Dot(axes=1)([weights, lstm])
        lstm_feat = layers.Reshape((512,))(lstm_ctx)

        # Fusion
        fused = layers.concatenate([cnn_feat, lstm_feat], axis=-1)
        x = layers.Dense(256)(fused)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


if __name__ == "__main__":
    print("[KERAS SUITE] Testing instantiation of all 10 Keras models...")
    for model_name in [
        "fasttext_baseline", "textcnn_standard", "textcnn_deep_multiscale",
        "bilstm_1layer_meanpool", "bilstm_2layer_concatpool", "bilstm_3layer_residual",
        "bilstm_additive_attention", "bigru_multihead_attention",
        "conv_bilstm_sequential", "dualstream_cnn_bilstm_fusion"
    ]:
        m = build_keras_model(model_name, vocab_size=5000, embedding_dim=300, num_classes=77)
        print(f"  ✓ {m.name:<30} | Total params: {m.count_params():,}")
    print("[KERAS SUITE] All 10 models instantiated successfully.")
