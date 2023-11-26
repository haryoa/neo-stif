import fire
from transformers import AutoTokenizer, BertForTokenClassification, BertConfig, BertForMaskedLM
from neo_stif.components.utils import create_label_map
import pandas as pd
from neo_stif.components.train_data_preparation import prepare_data_tagging_and_pointer
import datasets
from neo_stif.lit import LitTaggerOrInsertion
from torch.utils.data import DataLoader
from neo_stif.components.collator import FelixCollator, FelixInsertionCollator
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
from neo_stif.components.utils import compute_class_weights
from datasets import load_from_disk


def pointer(
    data_train,
    tokenizer,
    batch_size,
    with_validation=False,
    device="cuda",
    label_dict=None,
):
    rich_cb = RichProgressBar()
    pointer_network_config = BertConfig(
        vocab_size=len(label_dict) + 1,
        num_hidden_layers=2,
        num_attention_heads=1,
        pad_token_id=len(label_dict),
    )  # + 1 as the pad token
    lit_pointer = LitPointer(
        pointer_network_config,
        lr=LR_POINTER,
        num_classes=None
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="output/stif-i-f/felix-pointer/",
        filename="{epoch}-{val_loss:.2f}-{f1_val_step:.2f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
    )
    # ea_stop = EarlyStopping(patience=15, monitor="val_loss", mode="min")
    dev_dl = None

    if with_validation:
        df_dev = pd.read_csv("data/stif_indo/dev_with_pointing.csv")
        data_dev = datasets.Dataset.from_pandas(df_dev)
        data_dev, label_dict = prepare_data_tagging_and_pointer(
            data_dev, tokenizer, label_dict
        )
        dev_dl = DataLoader(
            data_dev,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=FelixCollator(tokenizer, pad_label_as_input=len(label_dict)),
        )

    train_dl = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=FelixCollator(tokenizer, pad_label_as_input=len(label_dict)),
    )
    trainer = Trainer(
        accelerator=device,
        devices=1,
        # val_check_interval=20,
        # check_val_every_n_epoch=None,
        callbacks=[rich_cb, checkpoint_callback],
    )
    trainer.fit(lit_pointer, train_dl, dev_dl)