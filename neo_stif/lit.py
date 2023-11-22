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

    def __init__(
        self,
        model,
        lr,
        num_classes=35,
        class_weight=None,
        tokenizer=None,
        label_dict=None,
        is_insertion=False,
        use_pointer=False,
        pointer_config=None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.class_weight = class_weight
        self.ce_loss = CrossEntropyLoss(class_weight)
        self.num_labels = num_classes
        # self.val_f1 = F1Score(
        #     "multiclass", average="macro", num_classes=num_classes, ignore_index=-100
        # )
        self.tokenizer = tokenizer
        self.label_dict = {j: i for i, j in label_dict.items()}
        self.is_insertion = is_insertion
        self.label_var_name = "labels" if is_insertion else "tag_labels"
        self.use_pointer = use_pointer
        self.pointer_model = (
            None
            if not use_pointer
            else PointerNetwork(
                pointer_config, previous_hidden_dim=model.config.hidden_size
            )
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def forward_pointer(self, input_ids, attention_mask, token_type_ids, previous_last_hidden, labels=None):
        return self.pointer_model(input_ids, attention_mask, token_type_ids, previous_last_hidden, labels=labels)


    def training_step(self, batch, batch_idx):
        input_to_model = {
            k: v
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        tag_pred = self(**input_to_model, output_hidden_states=True)
        last_hidden = tag_pred.hidden_states[-1]
        labels = batch[self.label_var_name]
        loss = self.ce_loss(tag_pred.logits.view(-1, self.num_labels), labels.view(-1))
        if self.use_pointer:
            input_to_pointer = {
                k: v
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "token_type_ids"]
            }
            input_to_pointer["input_ids"] = batch.pop("tag_labels_input")
            input_to_pointer['labels'] = batch["point_labels"]
            loss_pointer, last_att = self.forward_pointer(**input_to_pointer, previous_last_hidden=last_hidden)
            loss = loss + loss_pointer
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_to_model = {
            k: v
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        labels = batch[self.label_var_name]

        tag_pred = self(**input_to_model, labels=labels, output_hidden_states=True)
        last_hidden = tag_pred.hidden_states[-1]
        loss = tag_pred.loss

        if batch_idx == 0:
            if self.tokenizer is not None and self.label_dict is not None:
                input_ids = input_to_model["input_ids"][0]
                label = batch[self.label_var_name][0]
                ## reverse vocab
                reverse_vocab = {j: i for i, j in self.tokenizer.vocab.items()}
                input_ids_decoded = [reverse_vocab[x.cpu().item()] for x in input_ids]
                gold_label = [
                    reverse_vocab[z] if z != -100 else "IGNORED"
                    for z in label.cpu().numpy()
                ]
                print(
                    f"Input before going to output: {list(zip(input_ids_decoded, gold_label))}"
                )
                pred = tag_pred.logits[0].argmax(-1).detach().cpu().numpy()
                pred_label = [reverse_vocab[z] for z in pred]
                print(f"Input, pred: {list(zip(input_ids_decoded, pred_label))}")
            
        # self.val_f1(tag_pred.logits.argmax(-1), batch[self.label_var_name])
        # self.log("f1_val", self.val_f1, on_epoch=True, prog_bar=True)
        if self.use_pointer:
            input_to_pointer = {
                k: v
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "token_type_ids"]
            }
            input_to_pointer["input_ids"] = batch.pop("tag_labels_input")
            input_to_pointer['labels'] = batch["point_labels"]
            loss_pointer, _ = self.forward_pointer(**input_to_pointer, previous_last_hidden=last_hidden)
            loss = loss + loss_pointer
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_to_model = {
            k: v
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        tag_pred = self(**input_to_model, labels=batch[self.label_var_name])
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

    def __init__(self, config, lr, num_classes=None) -> None:
        super().__init__()
        self.model = PointerNetwork(config)
        self.lr = lr

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        input_to_model = {
            k: v
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        input_to_model["input_ids"] = batch.pop("tag_labels_input")

        tag_pred = self(**input_to_model, labels=batch["point_labels"])
        loss, last_att = tag_pred
        return loss

    def validation_step(self, batch, batch_idx):
        input_to_model = {
            k: v
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        input_to_model["input_ids"] = batch.pop("tag_labels_input")

        tag_pred = self(**input_to_model, labels=batch["point_labels"])
        loss, last_att = tag_pred
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_to_model = {
            k: v
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        input_to_model["input_ids"] = batch.pop("tag_labels_input")

        tag_pred = self(**input_to_model, labels=batch["point_labels"])
        loss, last_att = tag_pred
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
