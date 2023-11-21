

from lightning import LightningModule
from torch.optim import AdamW
from neo_stif.components.models import PointerNetwork


class LitTaggerOrInsertion(LightningModule):
    """
    Class Training Tagger or Insertion
    Insertion will be trained via MLM (other label than [MASK] will be -100)
    Tagger will be trained as TokenTagging
    """

    def __init__(self, model, lr) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        input_to_model = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        tag_pred = self(**input_to_model, labels=batch['tag_labels'])
        loss = tag_pred.loss
        return loss

    def validation_step(self, batch, batch_idx):
        input_to_model = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        tag_pred = self(**input_to_model, labels=batch['tag_labels'])
        loss = tag_pred.loss
        # self.val_f1(y_hat, y)
        # self.log("f1_val_step", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_to_model = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        tag_pred = self(**input_to_model, labels=batch['tag_labels'])
        loss = tag_pred.loss
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        return optimizer


class LitPointer(LightningModule):
    """
    Class Training Pointers
    """

    def __init__(self, config, lr) -> None:
        super().__init__()
        self.model = PointerNetwork(config)
        self.lr = lr
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        input_to_model = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        tag_pred = self(**input_to_model, labels=batch['tag_labels'])
        loss = tag_pred.loss
        return loss

    def validation_step(self, batch, batch_idx):
        input_to_model = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        tag_pred = self(**input_to_model, labels=batch['tag_labels'])
        loss = tag_pred.loss
        # self.val_f1(y_hat, y)
        # self.log("f1_val_step", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_to_model = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        tag_pred = self(**input_to_model, labels=batch['tag_labels'])
        loss = tag_pred.loss
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
