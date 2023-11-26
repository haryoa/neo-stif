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


def tagger(
    df_train,
    data_train,
    tokenizer,
    batch_size,
    model_path_or_name,
    with_validation=False,
    do_compute_class_weight=False,
    device="cuda",
    label_dict=None,
):
    rich_cb = RichProgressBar()

    checkpoint_callback = ModelCheckpoint(
        dirpath="output/stif-i-f/felix-tagger/",
        filename="{epoch}-{val_loss:.2f}-{f1_val_step:.2f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
    )
    ea_stop = EarlyStopping(patience=15, monitor="val_loss", mode="min")
    dev_dl = None
    class_weights = (
        compute_class_weights(df_train.label.apply(eval), num_classes=len(label_dict))
        if do_compute_class_weight
        else None
    )
    print(class_weights)
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

    pre_trained_bert = BertForTokenClassification.from_pretrained(
        model_path_or_name, num_labels=len(label_dict)
    )
    lit_tagger = LitTaggerOrInsertion(
        pre_trained_bert,
        lr=LR_TAGGER,
        num_classes=len(label_dict),
        class_weight=class_weights,
        tokenizer=tokenizer,
        label_dict=label_dict,
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
        val_check_interval=20,
        check_val_every_n_epoch=None,
        callbacks=[rich_cb, checkpoint_callback, ea_stop],
    )
    trainer.fit(lit_tagger, train_dl, dev_dl)