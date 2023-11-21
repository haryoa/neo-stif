from lightning import LightningModule
from torch.optim import AdamW
from neo_stif.components.models import PointerNetwork
from torchmetrics import F1Score
from torch.nn import CrossEntropyLoss


class LitTaggerOrInsertion(LightningModule):
    """
    Class Training Tagger or Insertion
    Insertion will be trained via MLM (other label than [MASK] will be -100)
    Tagger will be trained as TokenTagging
    """

    def __init__(self, model, lr, num_classes=35, class_weight=None, tokenizer=None, label_dict=None) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.class_weight = class_weight
        self.ce_loss = CrossEntropyLoss(class_weight)
        self.num_labels=num_classes
        self.val_f1 = F1Score('multiclass', average='macro', num_classes=num_classes, ignore_index=-100)
        self.tokenizer = tokenizer
        self.label_dict = {j:i for i,j in label_dict.items()}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        input_to_model = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        tag_pred = self(**input_to_model)
        labels=batch['tag_labels']
        loss = self.ce_loss(tag_pred.logits.view(-1, self.num_labels), labels.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_to_model = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        tag_pred = self(**input_to_model, labels=batch['tag_labels'])
        loss = tag_pred.loss
        
        if batch_idx == 0:
            if self.tokenizer is not None and self.label_dict is not None:
                input_ids = input_to_model['input_ids'][0]
                label = batch['tag_labels'][0]
                # get tokenizer inverse vocab
                vocab_reverse = {v: k for k, v in self.tokenizer.vocab.items()}
                input_ids_decoded = [vocab_reverse[x.cpu().item()] for x in input_ids]
                gold_label = [self.label_dict[z] if z != -100 else 'IGNORED' for z in label.cpu().numpy() ]
                print(f"Input before going to output: {list(zip(input_ids_decoded, gold_label))}")
                pred = tag_pred.logits[0].argmax(-1).detach().cpu().numpy()
                pred_label = [self.label_dict[z] for z in pred]
                print(f"Input, pred: {list(zip(input_ids_decoded, pred_label))}")

        self.val_f1(tag_pred.logits.argmax(-1), batch['tag_labels'])
        self.log("f1_val", self.val_f1, on_epoch=True, prog_bar=True)
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

    def __init__(self, config, lr, num_classes) -> None:
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
