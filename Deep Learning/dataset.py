"""
Banking77 Dataset, Vocabulary Builder, and Pretrained Embedding Utilities
========================================================================
Supports:
- Official Banking77 train (10,003 queries) and test (3,080 queries) split.
- Deterministic tokenization and vocabulary generation (Seed 42).
- Automatic GloVe-300d download, extraction, and embedding matrix construction.
- PyTorch Dataset & DataLoader generation.
"""

import os
import re
import zipfile
import urllib.request
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def clean_tokenize(text: str) -> List[str]:
    """
    Standard lowercased alphanumeric tokenizer for conversational banking queries.
    Handles contractions and punctuation boundaries cleanly.
    """
    text = text.lower().strip()
    # Separate contractions and punctuation
    text = re.sub(r"([?.!,¿'\"()-])", r" \1 ", text)
    tokens = [t for t in text.split() if t]
    return tokens


class Banking77Vocab:
    """
    Vocabulary mapping words to unique indices with PAD (0) and UNK (1).
    """
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}

    def build_vocab(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            tokens = clean_tokenize(text)
            counter.update(tokens)

        for word, freq in counter.most_common():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"[VOCAB] Built vocabulary with {len(self.word2idx):,} tokens (min_freq={self.min_freq}).")

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        tokens = clean_tokenize(text)
        indices = [self.word2idx.get(t, self.UNK_IDX) for t in tokens][:max_len]
        # Pad to max_len
        if len(indices) < max_len:
            indices = indices + [self.PAD_IDX] * (max_len - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)


class Banking77TorchDataset(Dataset):
    """
    PyTorch Dataset yielding (input_ids, label, length).
    """
    def __init__(self, texts: List[str], labels: List[int], vocab: Banking77Vocab, max_len: int = 64):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = self.vocab.encode(text, self.max_len)
        length = min(len(clean_tokenize(text)), self.max_len)
        if length == 0:
            length = 1
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long)
        }


def download_and_load_glove(
    cache_dir: str = "./glove_cache",
    embedding_dim: int = 300
) -> Dict[str, torch.Tensor]:
    """
    Downloads and extracts GloVe 6B embeddings if not present.
    Returns word -> 300d tensor dictionary.
    """
    os.makedirs(cache_dir, exist_ok=True)
    glove_txt = os.path.join(cache_dir, f"glove.6B.{embedding_dim}d.txt")
    glove_zip = os.path.join(cache_dir, "glove.6B.zip")

    if not os.path.exists(glove_txt):
        if not os.path.exists(glove_zip):
            url = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
            print(f"[GLOVE] Downloading GloVe 6B embeddings from {url}...")
            urllib.request.urlretrieve(url, glove_zip)
            print("[GLOVE] Download complete. Extracting...")
        with zipfile.ZipFile(glove_zip, "r") as zip_ref:
            zip_ref.extract(f"glove.6B.{embedding_dim}d.txt", cache_dir)
        print(f"[GLOVE] Extracted {glove_txt}")

    print(f"[GLOVE] Loading {glove_txt} into memory...")
    embeddings_dict = {}
    with open(glove_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            embeddings_dict[word] = vector
    print(f"[GLOVE] Successfully loaded {len(embeddings_dict):,} word vectors.")
    return embeddings_dict


def build_embedding_matrix(
    vocab: Banking77Vocab,
    glove_dict: Optional[Dict[str, torch.Tensor]],
    embedding_dim: int = 300
) -> torch.Tensor:
    """
    Constructs an [V, D] embedding matrix for the vocabulary using GloVe.
    Unknown tokens are initialized with N(0, 0.05).
    Padding token (idx 0) is set to 0.
    """
    matrix = torch.empty((len(vocab), embedding_dim), dtype=torch.float32)
    torch.nn.init.normal_(matrix, mean=0.0, std=0.05)
    matrix[vocab.PAD_IDX] = 0.0

    if glove_dict is not None:
        matched = 0
        for word, idx in vocab.word2idx.items():
            if idx in (vocab.PAD_IDX, vocab.UNK_IDX):
                continue
            if word in glove_dict:
                matrix[idx] = glove_dict[word]
                matched += 1
        coverage = (matched / (len(vocab) - 2)) * 100 if len(vocab) > 2 else 0
        print(f"[GLOVE COVERAGE] Matched {matched}/{len(vocab) - 2} tokens ({coverage:.2f}% coverage).")
    else:
        print("[EMBEDDINGS] Pretrained embeddings not provided. Initialized with random N(0, 0.05).")

    return matrix


def get_banking77_dataloaders(
    batch_size: int = 64,
    max_len: int = 64,
    cache_dir: str = "./glove_cache",
    embedding_dim: int = 300,
    download_glove: bool = True
) -> Tuple[DataLoader, DataLoader, Banking77Vocab, torch.Tensor, List[str]]:
    """
    Loads official full Banking77 splits (10,003 train, 3,080 test),
    builds vocabulary, loads GloVe embeddings, and returns train & test DataLoaders.
    """
    print("[DATA] Loading Banking77 dataset from HuggingFace...")
    ds = load_dataset("banking77")
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]
    label_names = ds["train"].features["label"].names

    print(f"[DATA] Train Samples: {len(train_texts):,} | Test Samples: {len(test_texts):,} | Classes: {len(label_names)}")

    vocab = Banking77Vocab(min_freq=1)
    vocab.build_vocab(train_texts)

    glove_dict = None
    if download_glove:
        try:
            glove_dict = download_and_load_glove(cache_dir, embedding_dim)
        except Exception as e:
            print(f"[WARNING] Could not load GloVe embeddings ({e}). Falling back to trainable random initialization.")

    embedding_matrix = build_embedding_matrix(vocab, glove_dict, embedding_dim)

    train_dataset = Banking77TorchDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = Banking77TorchDataset(test_texts, test_labels, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, vocab, embedding_matrix, label_names
