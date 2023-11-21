import fire
from transformers import AutoTokenizer, BertForTokenClassification, BertConfig
from neo_stif.components.utils import create_label_map
import pandas as pd
from neo_stif.components.train_data_preparation import prepare_data_tagging_and_pointer
import datasets
from neo_stif.lit import LitTaggerOrInsertion
from torch.utils.data import DataLoader
from neo_stif.components.collator import FelixCollator
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from neo_stif.components.utils import compute_class_weights


MAX_MASK = 35
USE_POINTING = True


model_dict = {
    'koto': "indolem/indobert-base-uncased"
}

def train(part: str = 'tagger', model='koto', batch_size=32, 
          with_validation=False, do_compute_class_weight=False):
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
    label_dict = create_label_map(MAX_MASK, USE_POINTING)
    rich_cb = RichProgressBar()

    df_train = pd.read_csv("data/stif_indo/train_with_pointing.csv")
    data_train = datasets.Dataset.from_pandas(df_train)
    data_train, label_dict = prepare_data_tagging_and_pointer(data_train, tokenizer, label_dict)
    model_path_or_name = model_dict[model]
    
    if part == 'tagger':
        dev_dl = None
        class_weights = compute_class_weights(df_train.label.apply(eval), num_classes=len(label_dict)) if do_compute_class_weight else None
        print(class_weights)
        if with_validation:
            df_dev = pd.read_csv("data/stif_indo/dev_with_pointing.csv")
            data_dev = datasets.Dataset.from_pandas(df_dev)
            data_dev, label_dict = prepare_data_tagging_and_pointer(data_dev, tokenizer, label_dict)
            dev_dl = DataLoader(data_dev, batch_size=batch_size, shuffle=True, collate_fn=FelixCollator(tokenizer, pad_label_as_input=len(label_dict)))

        pre_trained_bert = BertForTokenClassification.from_pretrained(model_path_or_name, num_labels=len(label_dict))
        lit_tagger = LitTaggerOrInsertion(pre_trained_bert, lr=1e-3, num_classes=len(label_dict), class_weight=class_weights)
        train_dl = DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=FelixCollator(tokenizer, pad_label_as_input=len(label_dict)))
        trainer = Trainer(accelerator='mps', devices=1, val_check_interval=20, check_val_every_n_epoch=None, callbacks=[rich_cb])
        trainer.fit(lit_tagger, train_dl, dev_dl)


if __name__ == '__main__':
    fire.Fire(train)