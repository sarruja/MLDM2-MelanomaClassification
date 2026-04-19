# =============================================================================
# train.py
# Zuständig für: Training starten, Callbacks konfigurieren
# Lesson 3: Early Stopping, Regularization
# =============================================================================

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,       # Training stoppen wenn keine Verbesserung mehr
    ModelCheckpoint,     # Bestes Modell speichern
    LearningRateMonitor  # Lernrate loggen
)
from lightning.pytorch.loggers import TensorBoardLogger

from datamodule import MelanomaDataModule
from model import MelanomaModel


# =============================================================================
# KONFIGURATION
# Alle Parameter hier zentral anpassen (kein Code durchsuchen nötig)
# =============================================================================

CONFIG = {
    # --- Pfade ---
    "csv_path":   "data/train.csv",         # Kaggle CSV mit Labels
    "image_dir":  "data/images/",           # Ordner mit den .jpg Bildern

    # --- Training ---
    "batch_size":    32,                    # Kleiner wenn GPU-Speicher voll
    "max_epochs":    30,                    # Maximum Epochen (Early Stopping stoppt früher)
    "learning_rate": 1e-4,                  # Adam Lernrate

    # --- Modell ---
    "pos_weight": 50.0,                     # Gewicht für Melanom-Klasse im Loss (~98/2)
    "dropout":    0.3,                      # Dropout-Rate (0.0 = kein Dropout)

    # --- Daten ---
    "val_split":  0.15,                     # 15% Validation
    "test_split": 0.15,                     # 15% Test
    "num_workers": 4,                       # Parallel-Prozesse (0 auf Windows falls Fehler)

    # --- Lokal testen ---
    # Auf True setzen um nur mit 10% der Daten zu trainieren (schnell zum Debuggen!)
    "fast_dev": False,
}


def main():
    # Reproduzierbarkeit sicherstellen (gleiche Zufallszahlen bei jedem Run)
    L.seed_everything(42)

    # -------------------------------------------------------
    # DATAMODULE initialisieren
    # Kümmert sich um: Laden, Splitten, Augmentieren, Batchen
    # -------------------------------------------------------
    datamodule = MelanomaDataModule(
        csv_path=CONFIG["csv_path"],
        image_dir=CONFIG["image_dir"],
        batch_size=CONFIG["batch_size"],
        val_split=CONFIG["val_split"],
        test_split=CONFIG["test_split"],
        num_workers=CONFIG["num_workers"],
    )

    # -------------------------------------------------------
    # MODELL initialisieren
    # EfficientNet-B0 mit vortrainierten ImageNet-Gewichten
    # -------------------------------------------------------
    model = MelanomaModel(
        learning_rate=CONFIG["learning_rate"],
        pos_weight=CONFIG["pos_weight"],
        dropout=CONFIG["dropout"],
    )

    # -------------------------------------------------------
    # CALLBACKS
    # Callbacks = Funktionen die Lightning automatisch aufruft
    # -------------------------------------------------------

    # Early Stopping: Training stoppen wenn val_loss sich nicht verbessert
    # (Lesson 3 - Early Stopping / Regularization)
    early_stopping = EarlyStopping(
        monitor="val_loss",   # beobachte Validation Loss
        patience=5,           # nach 5 Epochen ohne Verbesserung → stopp
        mode="min",           # Loss soll sinken
        verbose=True
    )

    # Model Checkpoint: Bestes Modell automatisch speichern
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="melanoma-{epoch:02d}-{val_auc:.3f}",
        monitor="val_auc",    # bestes Modell nach AUC auswählen
        mode="max",           # AUC soll steigen
        save_top_k=1,         # nur das beste Modell behalten
        verbose=True
    )

    # LR Monitor: Lernrate in den Logs anzeigen
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # -------------------------------------------------------
    # LOGGER: TensorBoard
    # Logs anschauen mit: tensorboard --logdir tb_logs/
    # -------------------------------------------------------
    logger = TensorBoardLogger("tb_logs", name="melanoma")

    # -------------------------------------------------------
    # TRAINER
    # Das Herzstück von Lightning – kümmert sich um:
    # - Training Loop (forward, backward, optimizer step)
    # - GPU/CPU Management
    # - Logging
    # - Callbacks aufrufen
    # -------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=CONFIG["max_epochs"],
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        logger=logger,
        accelerator="auto",   # automatisch GPU nutzen falls vorhanden, sonst CPU
        devices="auto",
        log_every_n_steps=10,

        # Nur 10% der Daten verwenden zum schnellen Testen (lokal debuggen)
        # Auf False setzen für echtes Training!
        fast_dev_run=CONFIG["fast_dev"],

        # Nützlich zum Debuggen: zeigt Modell-Summary und Batch-Shape
        # enable_model_summary=True,
    )

    # -------------------------------------------------------
    # TRAINING STARTEN
    # -------------------------------------------------------
    print("\n🚀 Training startet...\n")
    trainer.fit(model, datamodule=datamodule)

    # -------------------------------------------------------
    # EVALUATION AUF TEST SET
    # Erst nach dem Training, einmalig auf dem Test-Set!
    # (Lesson 1 - Train/Val/Test Split)
    # -------------------------------------------------------
    print("\n📊 Evaluation auf Test-Set...\n")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")  # bestes Checkpoint laden


if __name__ == "__main__":
    main()
