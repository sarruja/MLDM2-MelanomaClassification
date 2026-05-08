# =============================================================================
# model_baseline.py  –  Baseline Melanom-Klassifikationsmodell (nur Bilder)
#
# Was dieses File macht:
#   Gleiche Architektur wie model.py, ABER:
#   - Kein Metadata MLP
#   - Keine Fusion
#   - Nur EfficientNet-B0 → Classifier Head → Output
#
# Zweck: Vergleich mit dem multimodalen Modell
#   Baseline (nur Bilder) vs. Multimodal (Bilder + Metadaten)
#   → zeigt ob Metadaten wirklich etwas bringen
# =============================================================================

import torch
import torch.nn as nn
import lightning as L
import timm

from torchmetrics import AUROC, F1Score


class MelanomaModelBaseline(L.LightningModule):

    def __init__(self, learning_rate=1e-4, pos_weight=50.0, dropout=0.3):
        super().__init__()
        self.save_hyperparameters()

        # ------------------------------------------------------------------
        # BACKBONE: EfficientNet-B0 (identisch mit multimodalem Modell)
        # ------------------------------------------------------------------
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        backbone_out = self.backbone.num_features  # = 1280

        # ------------------------------------------------------------------
        # CLASSIFIER HEAD (nur Bild-Features, kein Metadaten-Input)
        # 1280 → 256 → 1  (statt 1312 → 256 → 1 beim multimodalen Modell)
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)
        )

        # Loss & Metriken (identisch mit multimodalem Modell)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.val_auc  = AUROC(task="binary")
        self.val_f1   = F1Score(task="binary")
        self.test_auc = AUROC(task="binary")
        self.test_f1  = F1Score(task="binary")

    def forward(self, images):
        # Kein Metadaten-Input!
        image_features = self.backbone(images)
        logits = self.classifier(image_features)
        return logits.squeeze(1)

    def training_step(self, batch, batch_idx):
        # Baseline ignoriert Metadaten komplett
        images, metadata, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, metadata, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels.float())
        probs = torch.sigmoid(logits)
        self.val_auc.update(probs, labels)
        self.val_f1.update(probs, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        f1  = self.val_f1.compute()
        self.log("val_auc", auc, prog_bar=True)
        self.log("val_f1",  f1,  prog_bar=True)
        self.val_auc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        images, metadata, labels = batch
        logits = self(images)
        probs = torch.sigmoid(logits)
        self.test_auc.update(probs, labels)
        self.test_f1.update(probs, labels)

    def on_test_epoch_end(self):
        auc = self.test_auc.compute()
        f1  = self.test_f1.compute()
        self.log("test_auc", auc)
        self.log("test_f1",  f1)
        print(f"\n✅ Baseline Test AUC-ROC : {auc:.4f}")
        print(f"✅ Baseline Test F1-Score : {f1:.4f}")
        self.test_auc.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }
