"""
Banking77 Dataset, Vocabulary Builder, and Pretrained Embedding Utilities
========================================================================
Supports:
- Official Banking77 train (10,003 queries) and test (3,080 queries) split.
- Resilient multi-tier loading (PolyAI/banking77, canonical GitHub CSVs).
- Deterministic tokenization and vocabulary generation (Seed 42).
- Automatic GloVe-300d download, extraction, and embedding matrix construction.
- PyTorch Dataset & DataLoader generation.
"""

import os
import re
import csv
import zipfile
import urllib.request
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader

# Official 77 Banking77 intent classes
BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits",
    "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
    "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal"
]

LABEL2ID = {name: i for i, name in enumerate(BANKING77_LABELS)}
ID2LABEL = {i: name for i, name in enumerate(BANKING77_LABELS)}


def clean_tokenize(text: str) -> List[str]:
    """
    Standard lowercased alphanumeric tokenizer for conversational banking queries.
    Handles contractions and punctuation boundaries cleanly.
    """
    text = text.lower().strip()
    text = re.sub(r"([?.!,¿'\"()-])", r" \1 ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def load_banking77_raw(cache_dir: str = "./data/banking77_official") -> Tuple[List[str], List[int], List[str], List[int], List[str]]:
    """
    Resilient loader for official full Banking77 splits (10,003 train, 3,080 test).
    1. Tries Hugging Face Hub `PolyAI/banking77`
    2. Tries Hugging Face Hub `banking77`
    3. Falls back to downloading canonical official CSVs directly from PolyAI's GitHub repository.
    """
    # Attempt 1: HuggingFace with namespace 'PolyAI/banking77'
    try:
        from datasets import load_dataset
        print("[DATA] Attempting HuggingFace load ('PolyAI/banking77')...")
        ds = load_dataset("PolyAI/banking77")
        tr_texts = ds["train"]["text"]
        tr_labels = ds["train"]["label"]
        te_texts = ds["test"]["text"]
        te_labels = ds["test"]["label"]
        label_names = ds["train"].features["label"].names
        if len(tr_texts) == 10003 and len(te_texts) == 3080:
            print(f"[DATA] Successfully loaded PolyAI/banking77 from HuggingFace.")
            return tr_texts, tr_labels, te_texts, te_labels, label_names
    except Exception as e1:
        print(f"[DATA NOTICE] Hugging Face PolyAI/banking77 load skipped ({type(e1).__name__}). Trying fallback...")

    # Attempt 2: Canonical GitHub CSVs from PolyAI repository
    os.makedirs(cache_dir, exist_ok=True)
    tr_csv = os.path.join(cache_dir, "train.csv")
    te_csv = os.path.join(cache_dir, "test.csv")

    def parse_csv(path):
        texts, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                texts.append(row["text"])
                cat = row.get("category") or row.get("label")
                labels.append(LABEL2ID[cat] if cat in LABEL2ID else int(cat))
        return texts, labels

    if not (os.path.exists(tr_csv) and os.path.exists(te_csv)):
        print("[DATA] Downloading canonical official splits directly from PolyAI GitHub repository...")
        train_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
        test_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
        for dst, url in [(tr_csv, train_url), (te_csv, test_url)]:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(dst, "wb") as out_f:
                out_f.write(resp.read())

    tr_texts, tr_labels = parse_csv(tr_csv)
    te_texts, te_labels = parse_csv(te_csv)
    assert len(tr_texts) == 10003, f"Expected 10,003 train queries, got {len(tr_texts)}"
    assert len(te_texts) == 3080, f"Expected 3,080 test queries, got {len(te_texts)}"
    print(f"[DATA] Successfully loaded canonical splits from {cache_dir}.")
    return tr_texts, tr_labels, te_texts, te_labels, BANKING77_LABELS


class Banking77Vocab:
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
        if len(indices) < max_len:
            indices = indices + [self.PAD_IDX] * (max_len - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)


class Banking77TorchDataset(Dataset):
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
    train_texts, train_labels, test_texts, test_labels, label_names = load_banking77_raw(
        cache_dir=os.path.join(cache_dir, "banking77_splits")
    )

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
