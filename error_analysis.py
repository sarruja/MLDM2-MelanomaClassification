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
                        choices=["multimodal", "baseline", "v2"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold für Klassifikation (default: 0.5)")
    parser.add_argument("--n_examples", type=int, default=8,
                        help="Count Beispielbilder pro Fehlertyp")
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
    elif model_type == "v2":
        from model_v2 import MelanomaModelV2
        model = MelanomaModelV2.load_from_checkpoint(checkpoint_path)
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
            if model_type in ["multimodal", "v2"]:
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
    fn_df = df[df["error_type"] == "FN"]
    fp_df = df[df["error_type"] == "FP"]
    tp_df = df[df["error_type"] == "TP"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.tight_layout()

    fig.suptitle("Error Analysis – Error Patterns", fontsize=18, fontweight="bold")

    # ---- 1. Body Site – FN und FP als gruppierte Bars ----
    site_col = "anatom_site_general_challenge"
    if site_col in df.columns:
        ax = axes[0]
        sites = df[site_col].fillna("unknown").unique()
        fn_counts = fn_df[site_col].fillna("unknown").value_counts().reindex(sites, fill_value=0)
        fp_counts = fp_df[site_col].fillna("unknown").value_counts().reindex(sites, fill_value=0)
        order = fn_counts.sort_values(ascending=False).index
        fn_counts = fn_counts[order]
        fp_counts = fp_counts[order]
        x = np.arange(len(order))
        width = 0.4
        ax.bar(x - width/2, fn_counts, width, label="FN (missed)", color="#E8735A", alpha=0.85)
        ax.bar(x + width/2, fp_counts, width, label="FP (false alarm)", color="#7F77DD", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=12)
        ax.set_title("Errors by Body Site", fontsize=15, fontweight="bold")
        ax.set_xlabel("Body Site", fontsize=13)
        ax.set_ylabel("Count", fontsize=13)
        ax.legend(fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.3)

    # ---- 2. Confidence-Verteilung ----
    ax = axes[1]
    for subset, label, color in [
        (fn_df, "FN (missed)", "#E8735A"),
        (fp_df, "FP (false alarm)", "#7F77DD"),
        (tp_df, "TP (correct)", "#1D9E75"),
    ]:
        if len(subset) > 0:
            ax.hist(subset["prob"], bins=15, alpha=0.6, label=label, color=color)
    ax.set_title("Model Confidence by Error Type", fontsize=15, fontweight="bold")
    ax.set_xlabel("Predicted Probability", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "error_patterns.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")  # entfernt Whitespace unten
    plt.close()
    print(f"Error Patterns gespeichert: {save_path}")

    # ---- Tabelle: FN nach Body Site ----
    site_col = "anatom_site_general_challenge"
    if site_col in df.columns:
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
    subset = df[df["error_type"] == error_type].copy()

    if len(subset) == 0:
        print(f"Keine {error_type} Beispiele gefunden.")
        return

    # Härteste Fehler auswählen
    if error_type == "FN":
        subset = subset.nsmallest(n, "prob")
        title = "False Negatives – Model was confident: no melanoma"
    else:
        subset = subset.nlargest(n, "prob")
        title = "False Positives – Model was confident: melanoma"

    # Nur 4 Beispiele anzeigen
    subset = subset.head(4)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    axes = axes.flatten()

    # Titel näher an Bilder
    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        y=0.95
    )

    image_dir = os.path.join(data_dir, "train")

    for i, (_, row) in enumerate(subset.iterrows()):
        img_path = os.path.join(image_dir, row["image_name"] + ".png")

        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, row["image_name"] + ".jpg")

        try:
            img = Image.open(img_path).convert("RGB")
            axes[i].imshow(img)
        except:
            axes[i].text(
                0.5, 0.5,
                "Not found",
                ha="center",
                va="center",
                transform=axes[i].transAxes
            )

        site = row.get("anatom_site_general_challenge", "unknown")
        sex  = row.get("sex", "?")
        age  = row.get("age_approx", "?")

        age_str = str(int(age)) if age != "?" else "?"

        # Grössere, besser lesbare Labels
        axes[i].text(
            0.98,
            0.98,
            f"p={row['prob']:.2f}\n{site}\n{sex}, {age_str}y",
            transform=axes[i].transAxes,
            fontsize=12,
            fontweight="normal",
            verticalalignment="top",
            horizontalalignment="right",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="black",
                alpha=0.55
            )
        )

        axes[i].axis("off")

    # Weniger Abstand zwischen Bildern
    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.88,
        bottom=0.02,
        wspace=0.04
    )

    save_path = os.path.join(
        output_dir,
        f"examples_{error_type.lower()}.png"
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Beispielbilder {error_type} gespeichert: {save_path}")

def combine_example_images(output_dir):
    """Lädt examples_fn.png und examples_fp.png und kombiniert sie nebeneinander."""
    fn_path = os.path.join(output_dir, "examples_fn.png")
    fp_path = os.path.join(output_dir, "examples_fp.png")

    fn_img = Image.open(fn_path)
    fp_img = Image.open(fp_path)

    # Gleiche Höhe sicherstellen
    height = max(fn_img.height, fp_img.height)
    fn_img = fn_img.resize((int(fn_img.width * height / fn_img.height), height))
    fp_img = fp_img.resize((int(fp_img.width * height / fp_img.height), height))

    # Nebeneinander zusammenfügen
    combined = Image.new("RGB", (fn_img.width + fp_img.width, height), color="white")
    combined.paste(fn_img, (0, 0))
    combined.paste(fp_img, (fn_img.width, 0))

    save_path = os.path.join(output_dir, "examples_combined.png")
    combined.save(save_path, dpi=(150, 150))
    print(f"Kombinierte Beispielbilder gespeichert: {save_path}")

def combine_example_images_vertical(output_dir):
    fn_path = os.path.join(output_dir, "examples_fn.png")
    fp_path = os.path.join(output_dir, "examples_fp.png")

    fn_img = Image.open(fn_path)
    fp_img = Image.open(fp_path)

    width = max(fn_img.width, fp_img.width)
    fn_img = fn_img.resize((width, int(fn_img.height * width / fn_img.width)))
    fp_img = fp_img.resize((width, int(fp_img.height * width / fp_img.width)))

    # Minimaler Abstand zwischen den zwei Bildern
    padding = 1
    combined = Image.new("RGB", (width, fn_img.height + fp_img.height + padding), color="white")
    combined.paste(fn_img, (0, 0))
    combined.paste(fp_img, (0, fn_img.height + padding))

    save_path = os.path.join(output_dir, "examples_combined_vertical.png")
    combined.save(save_path, dpi=(150, 150))
    print(f"Vertikale Beispielbilder gespeichert: {save_path}")


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
    combine_example_images(output_dir)
    combine_example_images_vertical(output_dir)
    
    print(f"\n✅ Error Analysis gespeichert in: results/{args.model}/error_analysis/")


if __name__ == "__main__":
    main()