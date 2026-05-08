# =============================================================================
# evaluate.py  –  Evaluation & Visualisierung
#
# Was dieses File macht:
#   1. Lädt ein trainiertes Modell (Multimodal oder Baseline)
#   2. Berechnet alle Metriken auf dem Test-Set
#   3. Exportiert TensorBoard Logs → training_history.csv (alle Epochen)
#   4. Speichert Metriken als JSON
#   5. Erstellt und speichert Plots:
#      - Training History (loss/auc über alle Epochen)
#      - ROC-Kurve
#      - Confusion Matrix
#      - Precision-Recall Kurve
#
# Usage:
#   python evaluate.py --model multimodal --checkpoint checkpoints/best-epoch=10-val_auc=0.8930.ckpt
#   python evaluate.py --model baseline   --checkpoint checkpoints/baseline/baseline-best-...ckpt
#
# Resultate in:
#   results/multimodal/  oder  results/baseline/
# =============================================================================

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # kein Display nötig (Server-Umgebung)

from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    f1_score, roc_auc_score
)
from torch.utils.tensorboard.writer import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from datamodule import MelanomaDataModule


# =============================================================================
# ARGPARSE
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Melanoma Model")
    parser.add_argument("--model", type=str, required=True,
                        choices=["multimodal", "baseline"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pfad zum .ckpt File")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


# =============================================================================
# TENSORBOARD EXPORT → CSV
# Liest alle Epochen-Metriken aus den TensorBoard Logs
# =============================================================================
def export_training_history(tb_log_dir, output_path):
    """
    Liest TensorBoard Logs und exportiert als CSV.
    tb_log_dir: z.B. "tb_logs/melanoma/version_0"
    """
    # Neueste version_X finden
    if not os.path.exists(tb_log_dir):
        print(f"TensorBoard Logs nicht gefunden: {tb_log_dir}")
        return None

    # Alle version_* Ordner finden, neueste nehmen
    versions = [d for d in os.listdir(tb_log_dir) if d.startswith("version_")]
    if not versions:
        print("Keine TensorBoard Versionen gefunden.")
        return None

    latest = sorted(versions)[-1]
    log_path = os.path.join(tb_log_dir, latest)
    print(f"Lese TensorBoard Logs: {log_path}")

    # EventAccumulator laden
    ea = EventAccumulator(log_path)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    print(f"Verfügbare Metriken: {available_tags}")

    # Alle Metriken in DataFrame laden
    data = {}
    for tag in available_tags:
        events = ea.Scalars(tag)
        steps  = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = pd.Series(values, index=steps)

    if not data:
        print("Keine Metriken in TensorBoard gefunden.")
        return None

    # Duplikate entfernen (z.B. test_auc wird nur einmal geloggt)
    # Nur Metriken die pro Epoche geloggt werden behalten
    epoch_metrics = ["train_loss", "val_loss", "val_auc", "val_f1"]
    data = {k: v for k, v in data.items() if k in epoch_metrics}

    # Duplikate im Index entfernen
    for key in data:
        data[key] = data[key][~data[key].index.duplicated(keep="last")]

    df = pd.DataFrame(data)
    df.index.name = "epoch"
    df.to_csv(output_path)
    print(f"Training History gespeichert: {output_path}")
    return df


# =============================================================================
# PREDICTIONS SAMMELN
# =============================================================================
def get_predictions(model_type, checkpoint_path, data_dir, batch_size):
    datamodule = MelanomaDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4
    )
    datamodule.setup()

    if model_type == "multimodal":
        from model import MelanomaModel
        model = MelanomaModel.load_from_checkpoint(checkpoint_path)
    else:
        from model_baseline import MelanomaModelBaseline
        model = MelanomaModelBaseline.load_from_checkpoint(checkpoint_path)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_probs  = []
    all_labels = []

    print("Sammle Predictions auf Test-Set...")
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            if model_type == "multimodal":
                images, metadata, labels = batch
                images   = images.to(device)
                metadata = metadata.to(device)
                logits   = model(images, metadata)
            else:
                images, metadata, labels = batch  # metadata wird ignoriert
                images = images.to(device)
                logits = model(images)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


# =============================================================================
# METRIKEN BERECHNEN
# =============================================================================
def compute_metrics(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auc_roc"          : float(roc_auc_score(labels, probs)),
        "f1_score"         : float(f1_score(labels, preds, zero_division=0)),
        "avg_precision"    : float(average_precision_score(labels, probs)),
        "threshold"        : threshold,
        "n_test_samples"   : int(len(labels)),
        "n_positive"       : int(labels.sum()),
        "n_negative"       : int((1 - labels).sum()),
        "positive_rate_pct": float(labels.mean() * 100),
        "true_negatives"   : int(tn),
        "false_positives"  : int(fp),
        "false_negatives"  : int(fn),
        "true_positives"   : int(tp),
        "sensitivity"      : float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "specificity"      : float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
    }
    return metrics, cm


# =============================================================================
# PLOTS
# =============================================================================
def plot_training_history(df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    if "train_loss" in df.columns:
        axes[0].plot(df.index, df["train_loss"], label="Train Loss", color="#7F77DD")
    if "val_loss" in df.columns:
        axes[0].plot(df.index, df["val_loss"], label="Val Loss", color="#E8735A")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUC
    if "val_auc" in df.columns:
        axes[1].plot(df.index, df["val_auc"], label="Val AUC-ROC", color="#1D9E75")
    if "val_f1" in df.columns:
        axes[1].plot(df.index, df["val_f1"], label="Val F1-Score", color="#F5A623")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation AUC-ROC & F1-Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Training History – Melanoma Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training History Plot gespeichert: {save_path}")


def plot_roc_curve(probs, labels, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#7F77DD", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1,
             label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve – Melanoma Classification", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC Curve gespeichert: {save_path}")


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Melanoma"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix – Melanoma Classification", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion Matrix gespeichert: {save_path}")


def plot_precision_recall(probs, labels, save_path):
    precision, recall, _ = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="#1D9E75", lw=2,
             label=f"PR Curve (AP = {avg_precision:.4f})")
    plt.axhline(y=labels.mean(), color="gray", linestyle="--", lw=1,
                label=f"Random (AP = {labels.mean():.4f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve – Melanoma Classification", fontsize=14)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Precision-Recall Curve gespeichert: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    output_dir = os.path.join("results", args.model)
    os.makedirs(output_dir, exist_ok=True)

    # ---- TensorBoard Export ----
    tb_name    = "melanoma" if args.model == "multimodal" else "melanoma_baseline"
    tb_log_dir = os.path.join("tb_logs", tb_name)
    history_df = export_training_history(
        tb_log_dir,
        os.path.join(output_dir, "training_history.csv")
    )

    # ---- Training History Plot ----
    if history_df is not None:
        plot_training_history(
            history_df,
            os.path.join(output_dir, "training_history.png")
        )

    # ---- Predictions auf Test-Set ----
    probs, labels = get_predictions(
        args.model, args.checkpoint, args.data_dir, args.batch_size
    )

    # ---- Metriken ----
    metrics, cm = compute_metrics(probs, labels)
    metrics["model_type"] = args.model
    metrics["checkpoint"] = args.checkpoint

    # Ausgabe
    print("\n" + "="*50)
    print(f"  RESULTATE – {args.model.upper()} MODEL")
    print("="*50)
    print(f"  AUC-ROC        : {metrics['auc_roc']:.4f}")
    print(f"  F1-Score       : {metrics['f1_score']:.4f}")
    print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
    print(f"  Sensitivity    : {metrics['sensitivity']:.4f}")
    print(f"  Specificity    : {metrics['specificity']:.4f}")
    print(f"  TP: {metrics['true_positives']}  FP: {metrics['false_positives']}  "
          f"TN: {metrics['true_negatives']}  FN: {metrics['false_negatives']}")
    print("="*50 + "\n")

    # JSON speichern
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metriken gespeichert: {output_dir}/metrics.json")

    # ---- Plots ----
    plot_roc_curve(probs, labels,
                   os.path.join(output_dir, "roc_curve.png"))
    plot_confusion_matrix(cm,
                          os.path.join(output_dir, "confusion_matrix.png"))
    plot_precision_recall(probs, labels,
                          os.path.join(output_dir, "precision_recall_curve.png"))

    print(f"\n✅ Alle Resultate gespeichert in: results/{args.model}/")


if __name__ == "__main__":
    main()
