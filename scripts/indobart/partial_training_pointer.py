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


MAX_MASK = 30
USE_POINTING = True


model_dict = {"koto": "indolem/indobert-base-uncased"}


LR_TAGGER = 5e-5 # due to the pre-trained nature
LR_POINTER = 1e-5 # no pre-trained
LR_INSERTION = 2e-5 # due to the pre-trained nature
VAL_CHECK_INTERVAL = 20


def train_stif(
    model_ckpt,
    model="koto",
    batch_size=32,
    with_validation=True, 
    do_compute_class_weight=False,
    device="cuda",
):
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
    label_dict = create_label_map(MAX_MASK, USE_POINTING)

    # Callback for trainer

    df_train = pd.read_csv("data/stif_indo/train_with_pointing.csv")
    data_train = datasets.Dataset.from_pandas(df_train)
    data_train, label_dict = prepare_data_tagging_and_pointer(
        data_train, tokenizer, label_dict
    )
    model_path_or_name = model_dict[model]

    rich_cb = RichProgressBar()

    checkpoint_callback = ModelCheckpoint(
        dirpath="output/stif-i-f/felix-tagger-pointer-continue/",
        filename="{epoch}-{val_loss:.2f}-{f1_val_step:.2f}",
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
    # print(class_weights)
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

    pointer_network_config = BertConfig(
        vocab_size=len(label_dict) + 1,
        num_hidden_layers=2,
        hidden_size=768,
        num_attention_heads=1,
        pad_token_id=len(label_dict),
    )  # + 1 as the pad token

    lit_tagger = LitTaggerOrInsertion.load_from_checkpoint(
        model_ckpt,
        model=pre_trained_bert,
        lr=5e-4,
        num_classes=len(label_dict),
        class_weight=class_weights,
        tokenizer=tokenizer,
        label_dict=label_dict,
        use_pointer=USE_POINTING,
        pointer_config=pointer_network_config,
        only_train_pointer=True,
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
        val_check_interval=50,
        max_epochs=500,
        check_val_every_n_epoch=None,
        callbacks=[rich_cb, checkpoint_callback],
    )
    trainer.fit(lit_tagger, train_dl, dev_dl)
    


if __name__ == "__main__":
    train_stif()
