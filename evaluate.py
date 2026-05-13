# =============================================================================
# evaluate.py  –  Evaluation & Visualisierung
#
# Was dieses File macht:
#   1. Lädt ein trainiertes Modell (Multimodal oder Baseline)
#   2. Findet automatisch den optimalen Threshold auf dem Val-Set
#      (maximiert F1-Score → besser als fixer 0.5 bei Klassenungleichgewicht)
#   3. Berechnet alle Metriken auf dem Test-Set mit optimalem Threshold
#   4. Exportiert TensorBoard Logs → training_history.csv (alle Epochen)
#   5. Speichert Metriken als JSON
#   6. Erstellt und speichert Plots:
#      - Training History (loss/auc über alle Epochen)
#      - ROC-Kurve mit optimalem Threshold markiert
#      - Confusion Matrix (default 0.5 UND optimaler Threshold)
#      - Precision-Recall Kurve
#
# Usage:
#   python evaluate.py --model multimodal --checkpoint checkpoints/best-epoch=15-val_auc=0.9087.ckpt
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
from PIL import Image
matplotlib.use("Agg")

from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    f1_score, roc_auc_score
)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from datamodule import MelanomaDataModule


# =============================================================================
# ARGPARSE
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Melanoma Model")
    parser.add_argument("--model", type=str, required=True,
                        choices=["multimodal", "baseline", "v2"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pfad zum .ckpt File")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


# =============================================================================
# TENSORBOARD EXPORT → CSV
# =============================================================================
def export_training_history(tb_log_dir, output_path):
    if not os.path.exists(tb_log_dir):
        print(f"TensorBoard Logs nicht gefunden: {tb_log_dir}")
        return None

    versions = [d for d in os.listdir(tb_log_dir) if d.startswith("version_")]
    if not versions:
        print("Keine TensorBoard Versionen gefunden.")
        return None

    latest   = sorted(versions)[-1]
    log_path = os.path.join(tb_log_dir, latest)
    print(f"Lese TensorBoard Logs: {log_path}")

    ea = EventAccumulator(log_path)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    print(f"Verfügbare Metriken: {available_tags}")

    data = {}
    for tag in available_tags:
        events = ea.Scalars(tag)
        steps  = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = pd.Series(values, index=steps)

    if not data:
        print("Keine Metriken in TensorBoard gefunden.")
        return None

    # Nur Epochen-Metriken behalten (test_auc etc. werden nur einmal geloggt)
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
# Gibt Predictions für Val-Set UND Test-Set zurück
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
    elif model_type == "v2":
        from model_v2 import MelanomaModelV2
        model = MelanomaModelV2.load_from_checkpoint(checkpoint_path)
    else:
        from model_baseline import MelanomaModelBaseline
        model = MelanomaModelBaseline.load_from_checkpoint(checkpoint_path)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    def collect(loader):
        probs_list  = []
        labels_list = []
        with torch.no_grad():
            for batch in loader:
                if model_type in ["multimodal", "v2"]:
                    images, metadata, labels = batch
                    images   = images.to(device)
                    metadata = metadata.to(device)
                    logits   = model(images, metadata)
                else:
                    images, metadata, labels = batch
                    images = images.to(device)
                    logits = model(images)
                probs_list.extend(torch.sigmoid(logits).cpu().numpy())
                labels_list.extend(labels.numpy())
        return np.array(probs_list), np.array(labels_list)

    print("Sammle Predictions auf Val-Set (für Threshold Tuning)...")
    val_probs, val_labels = collect(datamodule.val_dataloader())

    print("Sammle Predictions auf Test-Set...")
    test_probs, test_labels = collect(datamodule.test_dataloader())

    return val_probs, val_labels, test_probs, test_labels


# =============================================================================
# THRESHOLD TUNING
#
# Warum nicht einfach 0.5?
# Bei ~2% positiven Fällen ist das Modell "vorsichtig" — es braucht eine höhere
# Wahrscheinlichkeit bevor es Melanom sagt. Threshold 0.5 führt deshalb zu vielen
# False Negatives (verpasste Melanome).
#
# Wir berechnen drei Thresholds auf dem VAL-Set:
#   1. F1-optimal     → bester Kompromiss zwischen Precision und Recall
#   2. Sensitivity    → minimiert verpasste Melanome (wichtiger im medizin. Kontext!)
#
# WICHTIG: Threshold IMMER auf Val-Set bestimmen, nie auf Test-Set!
# Sonst würden wir den Test-Set "cheaten".
#
# Im medizinischen Kontext gilt:
#   False Negative (verpasstes Melanom) >> False Positive (falscher Alarm)
#   → Sensitivity-Threshold ist medizinisch relevanter als F1-Threshold
# =============================================================================
def find_optimal_thresholds(val_probs, val_labels, min_sensitivity=0.80):
    """
    Findet zwei optimale Thresholds auf dem Val-Set:
    1. F1-optimal: maximiert F1-Score
    2. Sensitivity-optimal: niedrigster Threshold der Sensitivity >= min_sensitivity erreicht
    """
    thresholds = np.arange(0.05, 0.95, 0.01)

    # --- F1-optimal ---
    best_f1_threshold = 0.5
    best_f1 = 0.0
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = t

    # --- Sensitivity-optimal ---
    # Höchsten Threshold finden der noch Sensitivity >= min_sensitivity erreicht
    # (höherer Threshold = weniger aber sicherere Vorhersagen)
    # Wir nehmen den niedrigsten Threshold der min_sensitivity erreicht
    sensitivity_threshold = 0.5
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        tp = int(((preds == 1) & (val_labels == 1)).sum())
        fn = int(((preds == 0) & (val_labels == 1)).sum())
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        if sensitivity >= min_sensitivity:
            sensitivity_threshold = t
            break  # niedrigsten Threshold nehmen der Ziel erreicht

    # Sensitivity bei diesem Threshold berechnen
    preds_sens = (val_probs >= sensitivity_threshold).astype(int)
    tp = int(((preds_sens == 1) & (val_labels == 1)).sum())
    fn = int(((preds_sens == 0) & (val_labels == 1)).sum())
    actual_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n📊 Threshold Tuning (auf Val-Set):")
    print(f"   Default Threshold          : 0.50")
    print(f"   F1-optimal Threshold       : {best_f1_threshold:.2f}  (Val F1: {best_f1:.4f})")
    print(f"   Sensitivity Threshold      : {sensitivity_threshold:.2f}  (Val Sensitivity: {actual_sensitivity:.4f})")
    print(f"   Ziel-Sensitivity           : >= {min_sensitivity}")
    print(f"\n   → Medizinisch empfohlen: Sensitivity-Threshold ({sensitivity_threshold:.2f})")
    print(f"     Verpasste Melanome werden minimiert auf Kosten von mehr False Positives")

    return float(best_f1_threshold), float(best_f1), float(sensitivity_threshold), float(actual_sensitivity)


# =============================================================================
# METRIKEN BERECHNEN
# =============================================================================
def compute_metrics(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auc_roc"          : float(roc_auc_score(labels, probs)),
        "f1_score"         : float(f1_score(labels, preds, zero_division=0)),
        "avg_precision"    : float(average_precision_score(labels, probs)),
        "threshold"        : float(threshold),
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

    if "train_loss" in df.columns:
        axes[0].plot(df.index, df["train_loss"], label="Train Loss", color="#7F77DD")
    if "val_loss" in df.columns:
        axes[0].plot(df.index, df["val_loss"], label="Val Loss", color="#E8735A")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

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


def plot_roc_curve(probs, labels, threshold, save_path):
    fpr, tpr, thresholds_roc = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Punkt auf der ROC-Kurve der dem optimalen Threshold entspricht
    idx = np.argmin(np.abs(thresholds_roc - threshold))

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#7F77DD", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1,
             label="Random (AUC = 0.5)")
    # Optimaler Threshold als Punkt markieren
    plt.scatter(fpr[idx], tpr[idx], color="#E8735A", s=100, zorder=5,
                label=f"Optimal threshold = {threshold:.2f}")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve – Melanoma Classification", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC Curve gespeichert: {save_path}")


def plot_confusion_matrix_comparison(probs, labels, threshold_default, threshold_optimal, save_path):
    """Zeigt zwei Confusion Matrices nebeneinander: default 0.5 vs. optimaler Threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, t, title in zip(
        axes,
        [threshold_default, threshold_optimal],
        [f"Default threshold = {threshold_default}", f"Optimal threshold = {threshold_optimal:.2f}"]
    ):
        preds = (probs >= t).astype(int)
        cm    = confusion_matrix(labels, preds)
        disp  = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Melanoma"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontsize=12)

    plt.suptitle("Confusion Matrix – Melanoma Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion Matrix gespeichert: {save_path}")


def plot_precision_recall(probs, labels, threshold, save_path):
    precision, recall, thresholds_pr = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)

    # Punkt für optimalen Threshold
    idx = np.argmin(np.abs(thresholds_pr - threshold))

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="#1D9E75", lw=2,
             label=f"PR Curve (AP = {avg_precision:.4f})")
    plt.scatter(recall[idx], precision[idx], color="#E8735A", s=100, zorder=5,
                label=f"Optimal threshold = {threshold:.2f}")
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


def threshold_comparison_table(test_probs, test_labels, save_path=None):
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    rows = []
    for t in thresholds:
        m, _ = compute_metrics(test_probs, test_labels, threshold=t)
        rows.append({
            "Threshold"  : t,
            "Sensitivity": round(m["sensitivity"], 3),
            "Specificity": round(m["specificity"], 3),
            "F1-Score"   : round(m["f1_score"], 3),
            "TP": m["true_positives"],
            "FN": m["false_negatives"],
            "FP": m["false_positives"],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def plot_error_analysis(model_type, checkpoint_path, data_dir, batch_size, threshold, save_path):
    """
    Zeigt je 4 False Positives und 4 False Negatives als Bildgrid.
    FP = Benign aber als Melanom klassifiziert
    FN = Melanom aber verpasst → medizinisch kritisch!
    """
    from datamodule import MelanomaDataset, MelanomaDataModule, preprocess_metadata
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Dataset neu aufbauen (ohne Augmentation, mit image_name)
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["target"], random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df["target"], random_state=42)

    train_df, _, age_min, age_max = preprocess_metadata(train_df)
    test_df,  _, _,       _       = preprocess_metadata(test_df, age_min, age_max)

    # Predictions sammeln + Index merken
    if model_type == "multimodal":
        from model import MelanomaModel
        model = MelanomaModel.load_from_checkpoint(checkpoint_path)
    elif model_type == "v2":
        from model_v2 import MelanomaModelV2
        model = MelanomaModelV2.load_from_checkpoint(checkpoint_path)
    else:
        from model_baseline import MelanomaModelBaseline
        model = MelanomaModelBaseline.load_from_checkpoint(checkpoint_path)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    from torchvision import transforms
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    from datamodule import MelanomaDataset
    test_dataset = MelanomaDataset(test_df, os.path.join(data_dir, "train"), transform=eval_transform)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images, metadata, labels = batch
            images = images.to(device)
            if model_type in ["multimodal", "v2"]:
                metadata = metadata.to(device)
                logits = model(images, metadata)
            else:
                logits = model(images)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)

    # FP / FN Indizes
    fp_idx = np.where((preds == 1) & (labels == 0))[0]
    fn_idx = np.where((preds == 0) & (labels == 1))[0]

    # Nach Konfidenz sortieren (interessanteste Fälle zuerst)
    fp_idx = fp_idx[np.argsort(probs[fp_idx])[::-1]][:4]  # höchste FP-Konfidenz
    fn_idx = fn_idx[np.argsort(probs[fn_idx])[::-1]][:4]  # höchste FN-Konfidenz (knapp verpasst)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Error Analysis (Threshold = {threshold:.2f})", fontsize=14)

    for col, idx in enumerate(fp_idx):
        row_data = test_df.iloc[idx]
        img_path = os.path.join(data_dir, "train", row_data["image_name"] + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(data_dir, "train", row_data["image_name"] + ".png")
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"FP | p={probs[idx]:.2f}", color="orange")
        axes[0, col].axis("off")

    for col, idx in enumerate(fn_idx):
        row_data = test_df.iloc[idx]
        img_path = os.path.join(data_dir, "train", row_data["image_name"] + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(data_dir, "train", row_data["image_name"] + ".png")
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        axes[1, col].imshow(img)
        axes[1, col].set_title(f"FN | p={probs[idx]:.2f}", color="red")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("False Positives\n(Benign → Melanoma)", fontsize=11, color="orange")
    axes[1, 0].set_ylabel("False Negatives\n(Missed Melanoma!)", fontsize=11, color="red")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Error Analysis gespeichert: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    output_dir = os.path.join("results", args.model)
    os.makedirs(output_dir, exist_ok=True)

    # ---- TensorBoard Export ----
    tb_name    = "melanoma" if args.model == "multimodal" else "melanoma_baseline" if args.model == "baseline" else "melanoma_v2"
    tb_log_dir = os.path.join("tb_logs", tb_name)
    history_df = export_training_history(
        tb_log_dir,
        os.path.join(output_dir, "training_history.csv")
    )
    if history_df is not None:
        plot_training_history(history_df,
                              os.path.join(output_dir, "training_history.png"))

    # ---- Predictions ----
    val_probs, val_labels, test_probs, test_labels = get_predictions(
        args.model, args.checkpoint, args.data_dir, args.batch_size
    )

    # ---- Threshold Tuning auf Val-Set ----
    f1_threshold, val_f1, sens_threshold, val_sensitivity = find_optimal_thresholds(
        val_probs, val_labels, min_sensitivity=0.80
    )

    # ---- Metriken mit allen drei Thresholds ----
    metrics_default, _  = compute_metrics(test_probs, test_labels, threshold=0.5)
    metrics_f1, _       = compute_metrics(test_probs, test_labels, threshold=f1_threshold)
    metrics_sens, cm_sens = compute_metrics(test_probs, test_labels, threshold=sens_threshold)

    # Ausgabe Vergleich
    print("\n" + "="*70)
    print(f"  RESULTATE – {args.model.upper()} MODEL")
    print("="*70)
    print(f"  {'Metrik':<20} {'Default (0.5)':>15} {'F1-optimal':>15} {'Sensitivity':>15}")
    print(f"  {'-'*65}")
    for key, label in [
        ("auc_roc",     "AUC-ROC"),
        ("f1_score",    "F1-Score"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("threshold",   "Threshold"),
        ("true_positives",  "TP"),
        ("false_negatives", "FN"),
        ("false_positives", "FP"),
    ]:
        fmt = ".4f" if key not in ["true_positives", "false_negatives", "false_positives", "threshold"] else ".2f" if key == "threshold" else "d"
        d = metrics_default[key]
        f = metrics_f1[key]
        s = metrics_sens[key]
        if fmt == "d":
            print(f"  {label:<20} {d:>15} {f:>15} {s:>15}")
        else:
            print(f"  {label:<20} {d:>15{fmt}} {f:>15{fmt}} {s:>15{fmt}}")
    print("="*70 + "\n")
    print("  → Medizinisch empfohlen: Sensitivity-Threshold")
    print(f"    Minimiert verpasste Melanome (FN: {metrics_sens['false_negatives']} vs {metrics_default['false_negatives']} bei default)\n")

    # JSON speichern (alle drei Thresholds)
    results = {
        "model_type" : args.model,
        "checkpoint" : args.checkpoint,
        "default_threshold"     : metrics_default,
        "f1_optimal_threshold"  : metrics_f1,
        "sensitivity_threshold" : metrics_sens,
        "val_f1_at_f1_threshold"         : val_f1,
        "val_sensitivity_at_sens_threshold": val_sensitivity,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metriken gespeichert: {output_dir}/metrics.json")

    # ---- Plots (mit Sensitivity-Threshold als Hauptthreshold) ----
    plot_roc_curve(test_probs, test_labels, sens_threshold,
                   os.path.join(output_dir, "roc_curve.png"))
    plot_confusion_matrix_comparison(test_probs, test_labels, 0.5, sens_threshold,
                                     os.path.join(output_dir, "confusion_matrix.png"))
    plot_precision_recall(test_probs, test_labels, sens_threshold,
                          os.path.join(output_dir, "precision_recall_curve.png"))

    plot_error_analysis(
    args.model, args.checkpoint, args.data_dir, args.batch_size,
    threshold=sens_threshold,
    save_path=os.path.join(output_dir, "error_analysis.png"))
    
    threshold_comparison_table(
    test_probs, test_labels,
    save_path=os.path.join(output_dir, "threshold_comparison.csv"))

    print(f"\n✅ Alle Resultate gespeichert in: results/{args.model}/")



if __name__ == "__main__":
    main()