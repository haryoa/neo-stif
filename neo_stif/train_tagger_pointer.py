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


def taggerpoint(
    df_train,
    data_train,
    tokenizer,
    batch_size,
    model_path_or_name,
    with_validation=False,
    do_compute_class_weight=False,
    device="cuda",
    label_dict=None,
    df_dev=None,
    use_pointing=True,
    output_dir_path="output/stif-i-f/felix-tagger-pointer/",
    lr=3e-5,
    hidden_size_pointer=64,
    num_hidden_layers_pointer=2,
    from_scratch=False,
    src_label="informal",
    tgt_label="formal",
):
    rich_cb = RichProgressBar()

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir_path,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )
    # ea_stop = EarlyStopping(patience=15, monitor="val_loss", mode="min")
    dev_dl = None
    class_weights = (
        compute_class_weights(df_train.label.apply(eval), num_classes=len(label_dict))
        if do_compute_class_weight
        else None
    )
    print(class_weights)
    if with_validation:
        data_dev = datasets.Dataset.from_pandas(df_dev)
        data_dev, label_dict = prepare_data_tagging_and_pointer(
            data_dev, tokenizer, label_dict
        )
        dev_dl = DataLoader(
            data_dev,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=FelixCollator(
                tokenizer,
                pad_label_as_input=len(label_dict),
                src=src_label,
                tgt=tgt_label,
            ),
        )

    if from_scratch:
        # copy config from model_path_or_name
        # and change the number of labels

        bert_config = BertConfig.from_pretrained(model_path_or_name)
        bert_config.num_labels = len(label_dict)
        pre_trained_bert = BertForTokenClassification(bert_config)  # from scratch
    else:
        pre_trained_bert = BertForTokenClassification.from_pretrained(
            model_path_or_name, num_labels=len(label_dict)
        )

    pointer_network_config = BertConfig(
        vocab_size=len(label_dict) + 1,
        num_hidden_layers=num_hidden_layers_pointer,
        hidden_size=hidden_size_pointer,
        num_attention_heads=1,
        pad_token_id=len(label_dict),
    )  # + 1 as the pad token

    lit_tagger = LitTaggerOrInsertion(
        pre_trained_bert,
        lr=lr,
        num_classes=len(label_dict),
        class_weight=class_weights,
        tokenizer=tokenizer,
        label_dict=label_dict,
        use_pointer=use_pointing,
        pointer_config=pointer_network_config,
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
        val_check_interval=30,
        max_epochs=500,
        precision=16,
        check_val_every_n_epoch=None,
        callbacks=[rich_cb, checkpoint_callback],
    )
    trainer.fit(lit_tagger, train_dl, dev_dl)
