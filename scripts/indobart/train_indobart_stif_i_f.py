from functools import partial
from transformers import MBartForConditionalGeneration
from indobenchmark import IndoNLGTokenizer
import pandas as pd
import datasets
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


MODEL_NAME = "indobenchmark/indobart-v2"
TRAIN_CSV = "data/stif_indo/train_with_pointing.csv"
VAL_CSV = "data/stif_indo/dev_with_pointing.csv"


def tokenize_function(examples, tokenizer, src="informal", tgt="formal"):
    src = examples[src]
    tgt = examples[tgt]

    src_tokenized = tokenizer(src, truncation=True)
    tgt_tokenized = tokenizer(tgt, truncation=True)["input_ids"]
    returned_dict = {
        "input_ids": src_tokenized["input_ids"],
        "attention_mask": src_tokenized["attention_mask"],
        "labels": tgt_tokenized,
    }
    return returned_dict


def main():
    tokenizer = IndoNLGTokenizer.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    df_train = pd.read_csv(TRAIN_CSV)
    df_train_used = df_train[["informal", "formal"]]

    df_val = pd.read_csv(VAL_CSV)
    df_val_used = df_val[["informal", "formal"]]

    df_train_data = datasets.Dataset.from_pandas(df_train_used)
    df_val_data = datasets.Dataset.from_pandas(df_val_used)

    train_tokenized = df_train_data.map(
        partial(tokenize_function, tokenizer=tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=["informal", "formal"],
    )
    val_tokenized = df_val_data.map(
        partial(tokenize_function, tokenizer=tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=["informal", "formal"],
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    early_stopping_cb = EarlyStoppingCallback(early_stopping_patience=5)

    training_args = TrainingArguments(
        learning_rate=1e-5,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        metric_for_best_model="eval_loss",
        save_strategy="steps",
        num_train_epochs=50,
        output_dir="outputs/stif-i-f/indobart-v2/",
        per_device_train_batch_size=8,
        save_total_limit=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_tokenized,
        eval_dataset = val_tokenized,
        tokenizer = tokenizer,
        data_collator = collator,
        callbacks = [early_stopping_cb]
    )
    
    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
