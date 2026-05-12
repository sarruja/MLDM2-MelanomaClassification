# =============================================================================
# hyperparameter_search.py  –  Hyperparameter Experimente für pos_weight
#
# Was dieses File macht:
#   Trainiert das multimodale Modell (v1) mit verschiedenen pos_weight Werten
#   und evaluiert jedes Modell automatisch.
#
# Experimente:
#   pos_weight = 25  → weniger Gewicht auf Melanome → mehr Specificity
#   pos_weight = 50  → Standard (Klassenungleichgewicht 98/2 ≈ 49)
#   pos_weight = 75  → mehr Gewicht auf Melanome → mehr Sensitivity
#
# Warum pos_weight variieren?
#   pos_weight kontrolliert den Trade-off zwischen Sensitivity und Specificity:
#   - Höheres pos_weight → Modell bestraft verpasste Melanome stärker
#                        → mehr True Positives, aber auch mehr False Positives
#   - Niedrigeres pos_weight → Modell ist konservativer
#                            → weniger False Positives, aber mehr verpasste Melanome
#
# Resultate in:
#   results/hparam_pos_weight_25/
#   results/hparam_pos_weight_50/  (bereits vorhanden von main training)
#   results/hparam_pos_weight_75/
#   results/hparam_comparison.csv  (Vergleichstabelle aller Experimente)
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from datamodule import MelanomaDataModule
from model import MelanomaModel
from evaluate import get_predictions, find_optimal_thresholds, compute_metrics


# =============================================================================
# KONFIGURATION
# =============================================================================
BASE_CONFIG = {
    "data_dir"      : "data",
    "batch_size"    : 32,
    "num_workers"   : 4,
    "learning_rate" : 1e-4,
    "dropout"       : 0.3,
    "max_epochs"    : 30,
    "fast_dev"      : False,
}

# Die drei Experimente
POS_WEIGHTS = [25, 50, 75]


# =============================================================================
# EINZELNES EXPERIMENT TRAINIEREN
# =============================================================================
def train_experiment(pos_weight, datamodule):
    """Trainiert ein Modell mit gegebenem pos_weight. Gibt Checkpoint-Pfad zurück."""

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: pos_weight = {pos_weight}")
    print(f"{'='*60}\n")

    L.seed_everything(42)  # Gleicher Seed für alle Experimente → fair vergleichen

    model = MelanomaModel(
        metadata_dim  = datamodule.meta_dim,
        learning_rate = BASE_CONFIG["learning_rate"],
        pos_weight    = float(pos_weight),
        dropout       = BASE_CONFIG["dropout"],
    )

    checkpoint_dir = f"checkpoints/hparam_pos_weight_{pos_weight}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_checkpoint = ModelCheckpoint(
        dirpath  = checkpoint_dir,
        filename = f"pos{pos_weight}-best-{{epoch:02d}}-{{val_auc:.4f}}",
        monitor  = "val_auc",
        mode     = "max",
        save_top_k = 1
    )
    last_checkpoint = ModelCheckpoint(
        dirpath  = checkpoint_dir,
        filename = "last",
        save_last = True
    )
    early_stopping = EarlyStopping(
        monitor="val_auc", mode="max", patience=5, verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = TensorBoardLogger("tb_logs", name=f"hparam_pos_weight_{pos_weight}")

    trainer = L.Trainer(
        max_epochs        = BASE_CONFIG["max_epochs"],
        callbacks         = [early_stopping, best_checkpoint, last_checkpoint, lr_monitor],
        logger            = logger,
        accelerator       = "auto",
        devices           = "auto",
        log_every_n_steps = 10,
        fast_dev_run      = BASE_CONFIG["fast_dev"],
    )

    trainer.fit(model, datamodule=datamodule)

    # Besten Checkpoint-Pfad zurückgeben
    best_ckpt = best_checkpoint.best_model_path
    print(f"\n✅ Bester Checkpoint: {best_ckpt}")
    return best_ckpt, trainer


# =============================================================================
# EXPERIMENT EVALUIEREN
# =============================================================================
def evaluate_experiment(pos_weight, checkpoint_path, datamodule):
    """Evaluiert ein trainiertes Modell und speichert Resultate."""

    output_dir = f"results/hparam_pos_weight_{pos_weight}"
    os.makedirs(output_dir, exist_ok=True)

    # Predictions
    val_probs, val_labels, test_probs, test_labels = get_predictions(
        "multimodal", checkpoint_path, BASE_CONFIG["data_dir"], BASE_CONFIG["batch_size"]
    )

    # Threshold Tuning
    f1_threshold, val_f1, sens_threshold, val_sensitivity = find_optimal_thresholds(
        val_probs, val_labels, min_sensitivity=0.80
    )

    # Metriken mit Default + Optimal Threshold
    metrics_default, _ = compute_metrics(test_probs, test_labels, threshold=0.5)
    metrics_sens, _    = compute_metrics(test_probs, test_labels, threshold=sens_threshold)

    # JSON speichern
    results = {
        "pos_weight"             : pos_weight,
        "checkpoint"             : checkpoint_path,
        "default_threshold"      : metrics_default,
        "sensitivity_threshold"  : metrics_sens,
        "sens_threshold_value"   : sens_threshold,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 pos_weight={pos_weight} | AUC: {metrics_default['auc_roc']:.4f} | "
          f"F1: {metrics_default['f1_score']:.4f} | "
          f"Sensitivity: {metrics_default['sensitivity']:.4f}")

    return metrics_default, metrics_sens


# =============================================================================
# VERGLEICHSTABELLE ERSTELLEN
# =============================================================================
def create_comparison_table(all_results):
    """Erstellt eine Vergleichstabelle aller Experimente."""

    rows = []
    for pos_weight, (metrics_default, metrics_sens) in all_results.items():
        rows.append({
            "pos_weight"        : pos_weight,
            "auc_roc"           : metrics_default["auc_roc"],
            "f1_default"        : metrics_default["f1_score"],
            "sensitivity_default": metrics_default["sensitivity"],
            "specificity_default": metrics_default["specificity"],
            "tp_default"        : metrics_default["true_positives"],
            "fn_default"        : metrics_default["false_negatives"],
            "fp_default"        : metrics_default["false_positives"],
        })

    df = pd.DataFrame(rows).sort_values("pos_weight")
    df.to_csv("results/hparam_comparison.csv", index=False)

    print("\n" + "="*70)
    print("  HYPERPARAMETER VERGLEICH – pos_weight")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    print(f"\n✅ Vergleichstabelle gespeichert: results/hparam_comparison.csv")

    return df


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs("results", exist_ok=True)

    # DataModule einmal aufsetzen (alle Experimente nutzen denselben Split)
    datamodule = MelanomaDataModule(
        data_dir   = BASE_CONFIG["data_dir"],
        batch_size = BASE_CONFIG["batch_size"],
        num_workers= BASE_CONFIG["num_workers"],
    )
    datamodule.setup()

    all_results = {}

    for pos_weight in POS_WEIGHTS:

        # pos_weight=50 haben wir bereits → überspringen wenn Checkpoint vorhanden
        existing = f"checkpoints/hparam_pos_weight_{pos_weight}"
        existing_ckpts = []
        if os.path.exists(existing):
            existing_ckpts = [f for f in os.listdir(existing) if f.endswith(".ckpt") and "best" in f]

        if existing_ckpts:
            checkpoint_path = os.path.join(existing, existing_ckpts[0])
            print(f"\n⏭️  pos_weight={pos_weight}: Checkpoint bereits vorhanden → überspringe Training")
            print(f"   Checkpoint: {checkpoint_path}")
        else:
            checkpoint_path, _ = train_experiment(pos_weight, datamodule)

        # Evaluieren
        metrics_default, metrics_sens = evaluate_experiment(pos_weight, checkpoint_path, datamodule)
        all_results[pos_weight] = (metrics_default, metrics_sens)

    # Vergleichstabelle
    create_comparison_table(all_results)


if __name__ == "__main__":
    main()
