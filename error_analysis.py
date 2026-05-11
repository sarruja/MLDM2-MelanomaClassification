# =============================================================================
# error_analysis.py  –  Analyse der falsch klassifizierten Bilder
#
# Was dieses File macht:
#   1. Lädt Modell-Predictions auf dem Test-Set
#   2. Identifiziert False Positives und False Negatives
#   3. Analysiert Muster nach:
#      - Körperstelle (body site)
#      - Geschlecht
#      - Alter
#      - Fehlende Metadaten
#   4. Speichert Beispielbilder der häufigsten Fehler
#   5. Speichert alle Ergebnisse als CSV + Plots
#
# Usage:
#   python error_analysis.py --model multimodal --checkpoint checkpoints/best-epoch=15-val_auc=0.9087.ckpt
#   python error_analysis.py --model baseline   --checkpoint checkpoints/baseline/baseline-best-epoch=06-val_auc=0.8921.ckpt
#
# Resultate in:
#   results/multimodal/error_analysis/
#   results/baseline/error_analysis/
# =============================================================================

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image
from sklearn.model_selection import train_test_split

from datamodule import MelanomaDataModule, preprocess_metadata


# =============================================================================
# ARGPARSE
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Error Analysis")
    parser.add_argument("--model", type=str, required=True,
                        choices=["multimodal", "baseline"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold für Klassifikation (default: 0.5)")
    parser.add_argument("--n_examples", type=int, default=8,
                        help="Anzahl Beispielbilder pro Fehlertyp")
    return parser.parse_args()


# =============================================================================
# TEST-SET PREDICTIONS MIT ORIGINAL CSV-DATEN
# Wir brauchen den originalen DataFrame (mit Alter, Geschlecht etc.)
# um die Fehler analysieren zu können
# =============================================================================
def get_predictions_with_metadata(model_type, checkpoint_path, data_dir, batch_size):
    """
    Gibt Predictions zurück zusammen mit dem originalen DataFrame
    (für Metadaten-Analyse der Fehler)
    """
    # ---- DataModule aufsetzen (gleicher Split wie beim Training!) ----
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))

    # Gleicher stratifizierter Split wie in datamodule.py
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["target"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["target"], random_state=42
    )

    # Metadaten preprocessen (gleiche Logik wie datamodule.py)
    train_df, _, age_min, age_max = preprocess_metadata(train_df)
    test_df, _, _, _ = preprocess_metadata(test_df, age_min, age_max)

    print(f"Test-Set: {len(test_df)} Samples | {test_df['target'].sum()} Melanome")

    # ---- DataModule für Predictions ----
    datamodule = MelanomaDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4
    )
    datamodule.setup()

    # ---- Modell laden ----
    if model_type == "multimodal":
        from model import MelanomaModel
        model = MelanomaModel.load_from_checkpoint(checkpoint_path)
    else:
        from model_baseline import MelanomaModelBaseline
        model = MelanomaModelBaseline.load_from_checkpoint(checkpoint_path)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # ---- Predictions sammeln ----
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            if model_type == "multimodal":
                images, metadata, labels = batch
                images   = images.to(device)
                metadata = metadata.to(device)
                logits   = model(images, metadata)
            else:
                images, metadata, labels = batch
                images = images.to(device)
                logits = model(images)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())

    test_df = test_df.reset_index(drop=True)
    test_df["prob"]  = all_probs
    test_df["label"] = all_labels

    return test_df


# =============================================================================
# FEHLER KLASSIFIZIEREN
# =============================================================================
def classify_errors(df, threshold):
    df = df.copy()
    df["pred"] = (df["prob"] >= threshold).astype(int)

    df["error_type"] = "correct"
    df.loc[(df["pred"] == 1) & (df["label"] == 0), "error_type"] = "FP"  # False Positive
    df.loc[(df["pred"] == 0) & (df["label"] == 1), "error_type"] = "FN"  # False Negative
    df.loc[(df["pred"] == 1) & (df["label"] == 1), "error_type"] = "TP"  # True Positive
    df.loc[(df["pred"] == 0) & (df["label"] == 0), "error_type"] = "TN"  # True Negative

    n_fp = (df["error_type"] == "FP").sum()
    n_fn = (df["error_type"] == "FN").sum()
    n_tp = (df["error_type"] == "TP").sum()
    n_tn = (df["error_type"] == "TN").sum()

    print(f"\n Klassifikation (threshold={threshold}):")
    print(f"   TP: {n_tp}  FP: {n_fp}  TN: {n_tn}  FN: {n_fn}")

    return df


# =============================================================================
# PATTERN ANALYSE
# Wo macht das Modell die meisten Fehler?
# =============================================================================
def analyze_patterns(df, output_dir):
    """Analysiert Fehlermuster nach Körperstelle, Geschlecht, Alter."""

    fn_df = df[df["error_type"] == "FN"]  # verpasste Melanome
    fp_df = df[df["error_type"] == "FP"]  # falscher Alarm
    tp_df = df[df["error_type"] == "TP"]  # korrekt erkannte Melanome

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Error Analysis – Fehlermuster", fontsize=15)

    # ---- 1. Body Site Verteilung ----
    site_col = "anatom_site_general_challenge"
    if site_col in df.columns:
        for ax, subset, title, color in zip(
            axes[0],
            [fn_df, fp_df, tp_df],
            ["False Negatives\n(verpasste Melanome)", "False Positives\n(falscher Alarm)", "True Positives\n(korrekt erkannt)"],
            ["#E8735A", "#7F77DD", "#1D9E75"]
        ):
            counts = subset[site_col].fillna("unknown").value_counts()
            counts.plot(kind="bar", ax=ax, color=color, alpha=0.8)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Body Site")
            ax.set_ylabel("Anzahl")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.3)

    # ---- 2. Altersverteilung ----
    age_col = "age_approx"
    if age_col in df.columns:
        ax = axes[1][0]
        for subset, label, color in [
            (fn_df, "FN (verpasst)", "#E8735A"),
            (fp_df, "FP (falscher Alarm)", "#7F77DD"),
            (tp_df, "TP (korrekt)", "#1D9E75"),
        ]:
            ages = subset[age_col].dropna()
            if len(ages) > 0:
                ax.hist(ages, bins=10, alpha=0.6, label=label, color=color)
        ax.set_title("Altersverteilung nach Fehlertyp")
        ax.set_xlabel("Alter")
        ax.set_ylabel("Anzahl")
        ax.legend()
        ax.grid(alpha=0.3)

    # ---- 3. Geschlechtsverteilung ----
    sex_col = "sex"
    if sex_col in df.columns:
        ax = axes[1][1]
        error_types = ["FN", "FP", "TP", "TN"]
        colors = ["#E8735A", "#7F77DD", "#1D9E75", "#888780"]
        x = np.arange(2)
        width = 0.2
        sexes = ["male", "female"]
        for i, (etype, color) in enumerate(zip(error_types, colors)):
            subset = df[df["error_type"] == etype]
            counts = [
                (subset["sex"] == s).sum() for s in sexes
            ]
            ax.bar(x + i * width, counts, width, label=etype, color=color, alpha=0.8)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(sexes)
        ax.set_title("Geschlecht nach Fehlertyp")
        ax.set_ylabel("Anzahl")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    # ---- 4. Confidence-Verteilung ----
    ax = axes[1][2]
    for subset, label, color in [
        (fn_df, "FN (verpasst)", "#E8735A"),
        (fp_df, "FP (falscher Alarm)", "#7F77DD"),
        (tp_df, "TP (korrekt)", "#1D9E75"),
    ]:
        if len(subset) > 0:
            ax.hist(subset["prob"], bins=15, alpha=0.6, label=label, color=color)
    ax.set_title("Modell-Confidence nach Fehlertyp")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Anzahl")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "error_patterns.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Error Patterns gespeichert: {save_path}")

    # ---- Tabelle: FN nach Body Site ----
    site_col = "anatom_site_general_challenge"
    if site_col in df.columns:
        print("\n📊 False Negatives nach Körperstelle:")
        fn_by_site = fn_df[site_col].fillna("unknown").value_counts()
        total_by_site = df[df["label"] == 1][site_col].fillna("unknown").value_counts()
        fn_rate = (fn_by_site / total_by_site * 100).round(1)
        summary = pd.DataFrame({
            "FN Count": fn_by_site,
            "Total Melanome": total_by_site,
            "FN Rate (%)": fn_rate
        }).sort_values("FN Rate (%)", ascending=False)
        print(summary.to_string())
        summary.to_csv(os.path.join(output_dir, "fn_by_body_site.csv"))

    return fn_df, fp_df


# =============================================================================
# BEISPIELBILDER SPEICHERN
# Zeigt die "härtesten" Fehler — wo das Modell am sichersten falsch war
# =============================================================================
def save_example_images(df, error_type, data_dir, output_dir, n=8):
    """
    Speichert n Beispielbilder des gegebenen Fehlertyps.
    Sortiert nach Confidence (härteste Fehler zuerst):
    - FN: höchste Probability (Modell war fast sicher, aber falsch)
    - FP: höchste Probability (Modell war sehr sicher dass es Melanom ist)
    """
    subset = df[df["error_type"] == error_type].copy()

    if len(subset) == 0:
        print(f"Keine {error_type} Beispiele gefunden.")
        return

    # Härteste Fehler zuerst
    if error_type == "FN":
        # FN: Modell hat niedrige Prob → sortiere nach niedrigster Prob (sicherste FN)
        subset = subset.nsmallest(n, "prob")
        title_suffix = "(Modell war sicher: kein Melanom)"
    else:
        # FP: Modell hat hohe Prob → sortiere nach höchster Prob
        subset = subset.nlargest(n, "prob")
        title_suffix = "(Modell war sicher: Melanom)"

    n_show = min(n, len(subset))
    fig, axes = plt.subplots(2, n_show // 2, figsize=(3 * (n_show // 2), 7))
    axes = axes.flatten()

    image_dir = os.path.join(data_dir, "train")

    for i, (_, row) in enumerate(subset.iterrows()):
        img_path = os.path.join(image_dir, row["image_name"] + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, row["image_name"] + ".jpg")

        try:
            img = Image.open(img_path).convert("RGB")
            axes[i].imshow(img)
        except:
            axes[i].text(0.5, 0.5, "Bild nicht\ngefunden",
                        ha="center", va="center", transform=axes[i].transAxes)

        site = row.get("anatom_site_general_challenge", "unknown")
        sex  = row.get("sex", "?")
        age  = row.get("age_approx", "?")
        axes[i].set_title(
            f"p={row['prob']:.2f}\n{site}\n{sex}, {age}y",
            fontsize=8
        )
        axes[i].axis("off")

    error_label = "False Negatives (verpasste Melanome)" if error_type == "FN" else "False Positives (falscher Alarm)"
    fig.suptitle(f"{error_label}\n{title_suffix}", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"examples_{error_type.lower()}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Beispielbilder {error_type} gespeichert: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    output_dir = os.path.join("results", args.model, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Predictions mit Metadaten ----
    df = get_predictions_with_metadata(
        args.model, args.checkpoint, args.data_dir, args.batch_size
    )

    # ---- Fehler klassifizieren ----
    df = classify_errors(df, threshold=args.threshold)

    # ---- CSV speichern (alle Predictions mit Metadaten) ----
    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    print(f"Predictions CSV gespeichert: {output_dir}/predictions.csv")

    # ---- Pattern Analyse ----
    fn_df, fp_df = analyze_patterns(df, output_dir)

    # ---- Beispielbilder ----
    print("\nSpeichere Beispielbilder...")
    save_example_images(df, "FN", args.data_dir, output_dir, n=args.n_examples)
    save_example_images(df, "FP", args.data_dir, output_dir, n=args.n_examples)

    print(f"\n✅ Error Analysis gespeichert in: results/{args.model}/error_analysis/")


if __name__ == "__main__":
    main()