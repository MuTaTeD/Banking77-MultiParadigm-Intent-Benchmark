#!/usr/bin/env python3
"""
Track E: Pre-Transformer Deep Learning Baseline Training & Evaluation Harness
=============================================================================
Trains and evaluates 10 pre-transformer architectures on the official full
Banking77 split (10,003 train / 3,080 test queries) with fixed seed 42.

Features:
- Automated GPU Peak Memory Telemetry (torch.cuda.max_memory_allocated).
- Automated Training & Evaluation Wall-Clock Timing.
- Resumable Execution (caches per-model metrics.json & model_best.pt).
- Soft-Voting Ensemble Experiment (loads saved checkpoints with ZERO re-training).
- JSON Predictions & Metrics export + Consolidated CSV summary table.
"""

import os
import sys
import json
import time
import random
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Local imports
from dataset import get_banking77_dataloaders
from models import MODEL_REGISTRY, get_model


def set_seed(seed: int = 42):
    """Enforces deterministic reproducibility across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """Formats elapsed seconds into standard Xm Ys format."""
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def evaluate(model: nn.Module, dataloader, device: torch.device):
    """
    Evaluates model on dataloader, computing:
    - Loss, Accuracy, Macro & Weighted F1, Macro & Weighted Precision / Recall
    - Raw predictions and targets
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            lengths = batch["length"].to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(all_targets) if all_targets else 0.0
    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    macro_p = precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    weighted_p = precision_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    macro_r = recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    weighted_r = recall_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0

    metrics = {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_f1": round(weighted_f1, 2),
        "macro_precision": round(macro_p, 2),
        "weighted_precision": round(weighted_p, 2),
        "macro_recall": round(macro_r, 2),
        "weighted_recall": round(weighted_r, 2),
    }
    return metrics, all_preds, all_targets


def get_model_probabilities(
    model_slug: str,
    vocab,
    model_dir: str,
    dataloader,
    device: torch.device
) -> Tuple[np.ndarray, List[int], int]:
    """
    Loads saved checkpoint weights (model_best.pt) without training
    and outputs softmax probability distribution [N, 77] across test set.
    """
    weights_path = os.path.join(model_dir, "model_best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    model = get_model(
        model_name=model_slug,
        vocab_size=len(vocab),
        embedding_dim=300,
        num_classes=77,
        pretrained_embeddings=None,
        freeze_embeddings=False
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            lengths = batch["length"].to(device)
            logits = model(input_ids, lengths)
            probs = F.softmax(logits, dim=-1).cpu()
            all_probs.append(probs)
            all_targets.extend(labels.cpu().numpy().tolist())

    return torch.cat(all_probs, dim=0).numpy(), all_targets, total_params


def run_soft_voting_ensemble(
    output_dir: str,
    test_loader,
    vocab,
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Executes Soft-Voting Ensembles by loading saved checkpoints with ZERO re-training:
    1. Ensemble_Top3_Trident: TextCNN_Standard + BiGRU_MultiHeadAttention + DualStream_CNN_BiLSTM_Fusion
    2. Ensemble_Top5_Deep: Top 5 non-transformer architectures
    """
    print("\n" + "="*80)
    print("[ENSEMBLE] Evaluating Soft-Voting Ensembles (Direct Checkpoint Loading)")
    print("="*80)

    candidate_slugs = [
        "textcnn_standard",
        "bigru_multihead_attention",
        "dualstream_cnn_bilstm_fusion",
        "textcnn_deep_multiscale",
        "bilstm_3layer_residual"
    ]

    prob_dict = {}
    param_dict = {}
    common_targets = None

    for slug in candidate_slugs:
        slug_dir = os.path.join(output_dir, slug)
        weights_file = os.path.join(slug_dir, "model_best.pt")
        if os.path.exists(weights_file):
            t0 = time.time()
            probs, targets, params = get_model_probabilities(slug, vocab, slug_dir, test_loader, device)
            prob_dict[slug] = probs
            param_dict[slug] = params
            common_targets = targets
            print(f"  ✓ Loaded '{slug}' ({params:,} params) -> Probabilities extracted in {time.time()-t0:.2f}s")
        else:
            print(f"  ✗ Checkpoint for '{slug}' not found in {slug_dir}. Skipping.")

    ensemble_configs = []
    top3 = ["textcnn_standard", "bigru_multihead_attention", "dualstream_cnn_bilstm_fusion"]
    if all(k in prob_dict for k in top3):
        ensemble_configs.append(("Ensemble_Top3_Trident", top3, "Kim-CNN + BiGRU-Attn + DualStream"))

    top5 = ["textcnn_standard", "bigru_multihead_attention", "dualstream_cnn_bilstm_fusion",
            "textcnn_deep_multiscale", "bilstm_3layer_residual"]
    if all(k in prob_dict for k in top5):
        ensemble_configs.append(("Ensemble_Top5_Deep", top5, "Top 5 Non-Transformer Suite"))

    ensemble_results = []
    for ens_name, members, desc in ensemble_configs:
        t0 = time.time()
        stacked = np.stack([prob_dict[m] for m in members], axis=0)
        avg_probs = np.mean(stacked, axis=0)
        preds = np.argmax(avg_probs, axis=1)
        eval_time = time.time() - t0

        acc = accuracy_score(common_targets, preds) * 100.0
        macro_f1 = f1_score(common_targets, preds, average="macro", zero_division=0) * 100.0
        weighted_f1 = f1_score(common_targets, preds, average="weighted", zero_division=0) * 100.0
        macro_p = precision_score(common_targets, preds, average="macro", zero_division=0) * 100.0
        macro_r = recall_score(common_targets, preds, average="macro", zero_division=0) * 100.0
        combined_params = sum(param_dict[m] for m in members)

        print(f"\n[ENSEMBLE RESULT: {ens_name}] ({desc})")
        print(f"   Accuracy: {acc:.2f}% | Macro F1: {macro_f1:.2f}% | Weighted F1: {weighted_f1:.2f}%")
        print(f"   Inference Time: {eval_time:.2f}s | Combined Parameters: {combined_params:,}")

        # Save predictions and metrics
        ens_dir = os.path.join(output_dir, ens_name.lower())
        os.makedirs(ens_dir, exist_ok=True)
        with open(os.path.join(ens_dir, "predictions.json"), "w") as f:
            json.dump({"targets": common_targets, "predictions": preds.tolist()}, f)

        res_dict = {
            "model_name": ens_name,
            "members": members,
            "total_params": combined_params,
            "accuracy": round(acc, 2),
            "macro_f1": round(macro_f1, 2),
            "weighted_f1": round(weighted_f1, 2),
            "macro_precision": round(macro_p, 2),
            "macro_recall": round(macro_r, 2),
            "inference_time_seconds": round(eval_time, 2)
        }
        with open(os.path.join(ens_dir, "metrics.json"), "w") as f:
            json.dump(res_dict, f, indent=2)

        ensemble_results.append({
            "Model Name": ens_name,
            "Parameters": combined_params,
            "Accuracy (%)": round(acc, 2),
            "Macro F1 (%)": round(macro_f1, 2),
            "Weighted F1 (%)": round(weighted_f1, 2),
            "Macro Precision (%)": round(macro_p, 2),
            "Macro Recall (%)": round(macro_r, 2),
            "Training Time": "N/A (Zero-Train Ensemble)",
            "Train Time (s)": 0.0,
            "Peak Train VRAM (GB)": 0.0,
            "Eval Time": format_time(eval_time),
            "Peak Eval VRAM (GB)": 0.0,
        })

    return ensemble_results


def train_single_model(
    model_slug: str,
    train_loader,
    test_loader,
    vocab,
    embedding_matrix,
    device: torch.device,
    output_dir: str,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    freeze_embeddings: bool = False
) -> Dict[str, Any]:
    """
    Trains and benchmarks a single architecture with automated timing & VRAM tracking.
    """
    model_dir = os.path.join(output_dir, model_slug)
    os.makedirs(model_dir, exist_ok=True)
    metrics_path = os.path.join(model_dir, "metrics.json")

    # Resumability check
    if os.path.exists(metrics_path):
        print(f"\n[CACHE] Model '{model_slug}' already trained. Loading {metrics_path}...")
        with open(metrics_path, "r") as f:
            return json.load(f)

    print(f"\n{'='*75}\n[TRAINING] Initializing Model: {model_slug}\n{'='*75}")
    set_seed(42)

    model = get_model(
        model_name=model_slug,
        vocab_size=len(vocab),
        embedding_dim=300,
        num_classes=77,
        pretrained_embeddings=embedding_matrix.clone(),
        freeze_embeddings=freeze_embeddings
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PARAMS] Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # Zero-Shot / Untrained baseline evaluation
    pre_metrics, _, _ = evaluate(model, test_loader, device)
    print(f"[PRE-TRAIN] Untrained Accuracy: {pre_metrics['accuracy']:.2f}% | Macro F1: {pre_metrics['macro_f1']:.2f}%")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Reset CUDA VRAM peak statistics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_start_time = time.time()
    best_test_acc = -1.0
    best_weights_path = os.path.join(model_dir, "model_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            lengths = batch["length"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(labels)

        scheduler.step()
        epoch_time = time.time() - t0
        avg_train_loss = epoch_loss / len(train_loader.dataset)

        # Quick test check
        cur_metrics, _, _ = evaluate(model, test_loader, device)
        cur_acc = cur_metrics["accuracy"]
        cur_f1 = cur_metrics["macro_f1"]

        if cur_acc > best_test_acc:
            best_test_acc = cur_acc
            torch.save(model.state_dict(), best_weights_path)
            saved_flag = " (Best Saved)"
        else:
            saved_flag = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test Acc: {cur_acc:.2f}% | Macro F1: {cur_f1:.2f}%{saved_flag}")

    train_total_seconds = time.time() - train_start_time
    train_peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0

    # Load best checkpoint for definitive post-evaluation
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=device))

    # Reset VRAM tracking for inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    eval_start_time = time.time()
    post_metrics, final_preds, targets = evaluate(model, test_loader, device)
    eval_total_seconds = time.time() - eval_start_time
    eval_peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0

    print(f"\n[FINAL BENCHMARK] {model_slug}:")
    print(f"   Accuracy: {post_metrics['accuracy']:.2f}% | Macro F1: {post_metrics['macro_f1']:.2f}% | Weighted F1: {post_metrics['weighted_f1']:.2f}%")
    print(f"   Training Time : {format_time(train_total_seconds)} ({train_total_seconds:.2f}s)")
    print(f"   Peak VRAM Train: {train_peak_vram_gb:.2f} GB")
    print(f"   Eval Time     : {format_time(eval_total_seconds)} ({eval_total_seconds:.2f}s)")
    print(f"   Peak VRAM Eval : {eval_peak_vram_gb:.2f} GB")

    result_summary = {
        "model_slug": model_slug,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "hardware_telemetry": {
            "training_time_seconds": round(train_total_seconds, 2),
            "training_time_formatted": format_time(train_total_seconds),
            "eval_time_seconds": round(eval_total_seconds, 2),
            "eval_time_formatted": format_time(eval_total_seconds),
            "peak_vram_training_gb": round(train_peak_vram_gb, 4),
            "peak_vram_eval_gb": round(eval_peak_vram_gb, 4),
        }
    }

    with open(metrics_path, "w") as f:
        json.dump(result_summary, f, indent=2)

    preds_path = os.path.join(model_dir, "predictions.json")
    with open(preds_path, "w") as f:
        json.dump({"targets": targets, "predictions": final_preds}, f)

    return result_summary


def main():
    parser = argparse.ArgumentParser(description="Track E: Pre-Transformer Deep Learning Suite")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "none"] + list(MODEL_REGISTRY.keys()),
                        help="Specific model to train, 'all' to run full suite, or 'none' to skip training.")
    parser.add_argument("--ensemble", action="store_true",
                        help="Run soft-voting ensemble using existing saved checkpoints.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Seed value (default: 42)")
    parser.add_argument("--output_dir", type=str, default="./results_deep_learning_full10k_seed42",
                        help="Directory to save checkpoints and metrics.")
    parser.add_argument("--glove_cache", type=str, default="./glove_cache",
                        help="Directory for GloVe embeddings download.")
    parser.add_argument("--freeze_embeddings", action="store_true",
                        help="Freeze GloVe embeddings during training.")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"[SETUP] GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"[SETUP] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Load Banking77 dataset & GloVe embeddings
    train_loader, test_loader, vocab, emb_matrix, label_names = get_banking77_dataloaders(
        batch_size=args.batch_size,
        cache_dir=args.glove_cache,
        download_glove=True
    )

    all_results = []
    if args.model != "none":
        models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
        for idx, slug in enumerate(models_to_run, 1):
            print(f"\n[{idx}/{len(models_to_run)}] Running Architecture: {slug}")
            res = train_single_model(
                model_slug=slug,
                train_loader=train_loader,
                test_loader=test_loader,
                vocab=vocab,
                embedding_matrix=emb_matrix,
                device=device,
                output_dir=args.output_dir,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                freeze_embeddings=args.freeze_embeddings
            )
            all_results.append(res)
    else:
        # Load all existing cached results from output_dir
        for slug in MODEL_REGISTRY.keys():
            m_file = os.path.join(args.output_dir, slug, "metrics.json")
            if os.path.exists(m_file):
                with open(m_file, "r") as f:
                    all_results.append(json.load(f))

    # Build Master Consolidated Table
    table_rows = []
    for r in all_results:
        post = r["post_metrics"]
        hw = r["hardware_telemetry"]
        table_rows.append({
            "Model Name": r.get("model_name", r.get("model_slug")),
            "Parameters": r["total_params"],
            "Accuracy (%)": post["accuracy"],
            "Macro F1 (%)": post["macro_f1"],
            "Weighted F1 (%)": post["weighted_f1"],
            "Macro Precision (%)": post["macro_precision"],
            "Macro Recall (%)": post["macro_recall"],
            "Training Time": hw["training_time_formatted"],
            "Train Time (s)": hw["training_time_seconds"],
            "Peak Train VRAM (GB)": hw["peak_vram_training_gb"],
            "Eval Time": hw["eval_time_formatted"],
            "Peak Eval VRAM (GB)": hw["peak_vram_eval_gb"],
        })

    # Run Soft-Voting Ensemble if requested or if running 'all'
    if args.ensemble or args.model == "all":
        ensemble_rows = run_soft_voting_ensemble(
            output_dir=args.output_dir,
            test_loader=test_loader,
            vocab=vocab,
            device=device
        )
        table_rows.extend(ensemble_rows)

    df = pd.DataFrame(table_rows)
    if "Accuracy (%)" in df.columns:
        df = df.sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(args.output_dir, "Track_E_Deep_Learning_Benchmark_Results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*80}\n[COMPLETE] All Track E experiments completed successfully!")
    print(f"[SUMMARY] Consolidated results saved to: {csv_path}\n{'='*80}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
