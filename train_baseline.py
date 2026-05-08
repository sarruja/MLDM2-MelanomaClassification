# =============================================================================
# train_baseline.py  –  Baseline Training (nur Bilder, keine Metadaten)
#
# Identisch mit train.py, aber verwendet MelanomaModelBaseline statt MelanomaModel
# Checkpoints werden in checkpoints/baseline/ gespeichert
# =============================================================================

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from datamodule import MelanomaDataModule
from model_baseline import MelanomaModelBaseline

CONFIG = {
    "data_dir"      : "data",
    "batch_size"    : 32,
    "num_workers"   : 4,
    "learning_rate" : 1e-4,
    "pos_weight"    : 50.0,
    "dropout"       : 0.3,
    "max_epochs"    : 30,
    "fast_dev"      : False,   # ← True = lokaler Test, False = echtes Training
}


def main():
    L.seed_everything(42)

    datamodule = MelanomaDataModule(
        data_dir   = CONFIG["data_dir"],
        batch_size = CONFIG["batch_size"],
        num_workers= CONFIG["num_workers"],
    )
    datamodule.setup()

    model = MelanomaModelBaseline(
        learning_rate = CONFIG["learning_rate"],
        pos_weight    = CONFIG["pos_weight"],
        dropout       = CONFIG["dropout"],
    )

    early_stopping = EarlyStopping(
        monitor="val_auc", mode="max", patience=5, verbose=True
    )
    best_checkpoint = ModelCheckpoint(
        dirpath="checkpoints/baseline/",
        filename="baseline-best-{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1
    )
    last_checkpoint = ModelCheckpoint(
        dirpath="checkpoints/baseline/",
        filename="last",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = TensorBoardLogger("tb_logs", name="melanoma_baseline")

    resume_from = None
    # resume_from = "checkpoints/baseline/last.ckpt"  # ← einkommentieren zum Fortsetzen

    trainer = L.Trainer(
        max_epochs        = CONFIG["max_epochs"],
        callbacks         = [early_stopping, best_checkpoint, last_checkpoint, lr_monitor],
        logger            = logger,
        accelerator       = "auto",
        devices           = "auto",
        log_every_n_steps = 10,
        fast_dev_run      = CONFIG["fast_dev"],
    )

    print("\n🚀 Baseline Training startet...\n")
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from)

    if not CONFIG["fast_dev"]:
        print("\n📊 Baseline Evaluation auf Test-Set...\n")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
