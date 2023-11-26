import fire
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    BertConfig,
    BertForMaskedLM,
)
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


def insertion(
    processed_train_data,
    processed_dev_data,
    tokenizer,
    batch_size,
    model_path_or_name,
    device="cuda",
    label_dict=None,
    output_dir_path="output/stif-i-f/felix-insert/",
    lr=2e-5,
    from_scratch=False,
):
    """
    Perform insertion task using the given parameters.

    Args:
        processed_train_data (str): Filepath to the processed training data.
        processed_dev_data (str): Filepath to the processed development data.
        tokenizer: Tokenizer object used for tokenizing the data.
        batch_size (int): Number of samples per batch.
        model_path_or_name (str): Path or name of the pre-trained model.
        device (str, optional): Device to use for training. Defaults to "cuda".
        label_dict (dict, optional): Dictionary mapping labels to their indices. Defaults to None.

    Returns:
        None
    """

    rich_cb = RichProgressBar()
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir_path,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )
    # ea_stop = EarlyStopping(patience=5, monitor="val_loss", mode="min")
    train_data = load_from_disk(processed_train_data)
    dev_data = load_from_disk(processed_dev_data)
    train_dl = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=FelixInsertionCollator(tokenizer),
    )
    dev_dl = DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=FelixInsertionCollator(tokenizer),
    )
    if from_scratch:
        config = BertConfig.from_pretrained(model_path_or_name)
        config.num_labels = len(label_dict)
        model = BertForMaskedLM(config)
    else:
        model = BertForMaskedLM.from_pretrained(model_path_or_name)
    lit_insert = LitTaggerOrInsertion(
        model,
        lr=lr,
        num_classes=model.config.vocab_size,
        class_weight=None,
        tokenizer=tokenizer,
        label_dict=label_dict,
        is_insertion=True,
    )
    trainer = Trainer(
        accelerator=device,
        devices=1,
        max_epochs=500,
        callbacks=[rich_cb, checkpoint_callback],
    )
    trainer.fit(lit_insert, train_dl, dev_dl)
