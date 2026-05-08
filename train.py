# =============================================================================
# train.py  –  Training starten
#
# Hier passiert: DataModule + Model zusammenbauen, Trainer konfigurieren,
# Training starten, Test-Evaluation durchführen.
#
# Zum lokalen Testen:   fast_dev = True  (nur 1 Batch, Sekunden)
# Für echtes Training:  fast_dev = False (auf LightningAI mit GPU!)
# =============================================================================

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from datamodule import MelanomaDataModule
from model import MelanomaModel

# =============================================================================
# KONFIGURATION – alle Hyperparameter an einem Ort
# =============================================================================
CONFIG = {
    "data_dir"      : "data",
    "batch_size"    : 32,
    "num_workers"   : 4,       # Windows: auf 0 setzen falls Fehler!
    "learning_rate" : 1e-4,
    "pos_weight"    : 50.0,    # ~98/2 Klassenungleichgewicht
    "dropout"       : 0.3,
    "max_epochs"    : 30,
    "fast_dev"      : True,    # ← True = lokaler Test (1 Batch), False = echtes Training
}


def main():
    L.seed_everything(42)  # Reproduzierbarkeit

    # ---- DataModule ----
    datamodule = MelanomaDataModule(
        data_dir   = CONFIG["data_dir"],
        batch_size = CONFIG["batch_size"],
        num_workers= CONFIG["num_workers"],
    )

    # setup() aufrufen damit meta_dim bekannt ist bevor wir das Modell bauen
    datamodule.setup()

    # ---- Modell ----
    # meta_dim kommt direkt vom DataModule (9 Features: Alter + Geschlecht + Körperstelle)
    model = MelanomaModel(
        metadata_dim  = datamodule.meta_dim,
        learning_rate = CONFIG["learning_rate"],
        pos_weight    = CONFIG["pos_weight"],
        dropout       = CONFIG["dropout"],
    )

    # ---- Callbacks ----
    # Early Stopping: stoppt Training wenn val_auc sich 5 Epochen nicht verbessert
    early_stopping = EarlyStopping(
        monitor="val_auc", mode="max", patience=5, verbose=True
    )

    # Model Checkpoint: speichert nur das beste Modell (nach val_auc)
    checkpoint = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best-{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1
    )

    # LR Monitor: loggt die Lernrate → in TensorBoard sichtbar
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ---- Logger ----
    # TensorBoard anschauen mit: tensorboard --logdir tb_logs/
    logger = TensorBoardLogger("tb_logs", name="melanoma")

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs   = CONFIG["max_epochs"],
        callbacks    = [early_stopping, checkpoint, lr_monitor],
        logger       = logger,
        accelerator  = "auto",   # GPU falls vorhanden, sonst CPU
        devices      = "auto",
        log_every_n_steps = 10,
        fast_dev_run = CONFIG["fast_dev"],  # True = schneller lokaler Test
    )

    # ---- Training ----
    print("\n🚀 Training startet...\n")
    trainer.fit(model, datamodule=datamodule)

    # ---- Test-Evaluation ----
    # Nur beim echten Training (fast_dev hat kein Checkpoint gespeichert)
    if not CONFIG["fast_dev"]:
        print("\n📊 Evaluation auf Test-Set...\n")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
