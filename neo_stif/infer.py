import fire
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    BertConfig,
    BertForMaskedLM,
)
from neo_stif.components.infer_utils import (
    beam_search_single_tagging,
    realize_beam_search,
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
import torch
from pprint import pprint
from tqdm import tqdm
import numpy as np

MAX_MASK = 30
USE_POINTING = True


model_dict = {"koto": "indolem/indobert-base-uncased"}


LR_TAGGER = 5e-5  # due to the pre-trained nature
LR_POINTER = 1e-5  # no pre-trained
LR_INSERTION = 2e-5  # due to the pre-trained nature
VAL_CHECK_INTERVAL = 20


def generate_felix(
    test_csv_path,
    out_csv_path,
    ckpt_tagger_path,
    ckpt_insertion_path,
    with_class_weight=False,
):
    """
    Perform felix inference task using the given parameters.

    Args:
        test_csv_path (str): Filepath to the test data.
        out_csv_path (str): Filepath to the output data.
        ckpt_tagger_path (str): Filepath to the tagger checkpoint.
        ckpt_insertion_path (str): Filepath to the insertion checkpoint.
        with_class_weight (bool, optional): Whether to use class weight. Defaults to False.

    Returns:
        None
    """
    model_path_or_name = model_dict["koto"]
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    label_dict = create_label_map(MAX_MASK, USE_POINTING)

    # Callback for trainer

    df_test = pd.read_csv(test_csv_path)
    data_test = datasets.Dataset.from_pandas(df_test)
    data_test, label_dict = prepare_data_tagging_and_pointer(
        data_test, tokenizer, label_dict
    )
    pre_trained_bert = BertForTokenClassification.from_pretrained(
        model_path_or_name, num_labels=len(label_dict)
    )
    pointer_network_config = BertConfig(
        vocab_size=len(label_dict) + 1,
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=1,
        pad_token_id=len(label_dict),
    )  # + 1 as the pad token

    class_weight = (
        compute_class_weights(data_test["tag_labels"], len(label_dict))
        if with_class_weight
        else None
    )

    lit_tagger = LitTaggerOrInsertion.load_from_checkpoint(
        ckpt_tagger_path,
        model=pre_trained_bert,
        lr=2e-5,
        num_classes=len(label_dict),
        class_weight=class_weight,
        tokenizer=tokenizer,
        label_dict=label_dict,
        use_pointer=USE_POINTING,
        pointer_config=pointer_network_config,
        map_location=torch.device("cpu"),
    )

    pre_trained_another_bert = BertForMaskedLM.from_pretrained(model_path_or_name)

    lit_insert = LitTaggerOrInsertion.load_from_checkpoint(
        ckpt_insertion_path,
        model=pre_trained_another_bert,
        lr=LR_INSERTION,
        num_classes=pre_trained_another_bert.config.vocab_size,
        class_weight=None,
        tokenizer=tokenizer,
        label_dict=label_dict,
        is_insertion=True,
        map_location=torch.device("cpu"),
    )

    lit_tagger = lit_tagger.eval()
    lit_tagger.freeze()
    tokenizer_vocab_reverse = {v: k for k, v in tokenizer.vocab.items()}
    # label_dict

    # reverese the label dict
    label_dict_reverse = {v: k for k, v in label_dict.items()}
    deleted_tags = ["DELETE", "PAD_TAG", "PAD"]
    results = []
    fail_count = 0
    for data in tqdm(data_test):
        with torch.no_grad():
            inp_to_model = tokenizer(data["informal"], return_tensors="pt").to("cpu")
            out_logits = lit_tagger.forward(**inp_to_model, output_hidden_states=True)
            # decoded_seq = [
            #     tokenizer_vocab_reverse[x.item()] for x in inp_to_model["input_ids"][0]
            # ]
            # decoded_label = [
            #     label_dict_reverse[x.item()] for x in out_logits.logits.argmax(-1)[0]
            # ]
            inp_tag = torch.LongTensor([data["tag_labels"]])
            _, out_att = lit_tagger.forward_pointer(
                input_ids=inp_tag,
                attention_mask=inp_to_model["attention_mask"],
                token_type_ids=inp_to_model["token_type_ids"],
                previous_last_hidden=out_logits.hidden_states[-1],
            )
            # att_output = out_att.argmax(-1)
            # pprint(
            #     list(
            #         zip(
            #             list(range(len(decoded_seq))),
            #             decoded_seq,
            #             decoded_label,
            #             att_output[0][0].numpy(),
            #             data["point_labels"],
            #         )
            #     )
            # )
            tagger_logit, pointer_att = out_logits.logits.numpy(), out_att
            input_word_ids = inp_to_model["input_ids"][0].numpy()
            last_token_index = (
                inp_to_model["input_ids"][0].tolist().index(tokenizer.vocab["[SEP]"])
            )
            predicted_tags = list(np.argmax(tagger_logit, axis=-1))[0]
            non_deleted_indexes = set(
                i
                for i, tag in enumerate(predicted_tags[: last_token_index + 1])
                if label_dict_reverse[int(tag)] not in deleted_tags
            )
            source_tokens = [
                tokenizer_vocab_reverse[x.item()] for x in inp_to_model["input_ids"][0]
            ]
            sep_indexes = set(
                [
                    i
                    for i, token in enumerate(source_tokens)
                    if token == "[SEP]" and i in non_deleted_indexes
                ]
            )
            pointer_np = pointer_att[0][0].numpy()
            try:
                best_sequence = beam_search_single_tagging(
                    list(pointer_np),
                    non_deleted_indexes,
                    sep_indexes,
                    4,
                    last_token_index,
                    100,
                )

                realized_inp_insertion = realize_beam_search(
                    input_word_ids,
                    best_sequence,
                    predicted_tags,
                    last_token_index + 1,
                    label_dict_reverse,
                    tokenizer,
                )

                input_ids = tokenizer.convert_tokens_to_ids(realized_inp_insertion)
                attention_mask = [1] * len(input_ids)
                token_type_ids = [0] * len(input_ids)

                # make them to torch
                input_ids = torch.LongTensor([input_ids])
                attention_mask = torch.LongTensor([attention_mask])
                token_type_ids = torch.LongTensor([token_type_ids])

                with torch.no_grad():
                    out = lit_insert.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    out_label = out.logits.argmax(-1)[0].numpy()
                input_ids_detokenized = tokenizer.convert_ids_to_tokens(
                    input_ids[0].numpy()
                )
                out_ids_detokenized = tokenizer.convert_ids_to_tokens(out_label)

                list(zip(input_ids_detokenized, out_ids_detokenized))
                # only change [MASK] from input_ids_detokenized

                out = [
                    out_ids_detokenized[i] if x == "[MASK]" else x
                    for i, x in enumerate(input_ids_detokenized)
                ]

                # remove sequence between [UNK] and [PAD].
                out_seq = []

                is_permited = True
                for chr in out:
                    if chr == "[UNK]":
                        is_permited = False
                    if chr == "[PAD]":
                        is_permited = True
                        continue
                    if is_permited:
                        out_seq.append(chr)

                results.append(
                    tokenizer.decode(
                        tokenizer.convert_tokens_to_ids(out_seq), skip_special_tokens=True
                    )
                )
            except:
                results.append(data["informal"])
                fail_count += 1
                continue
    print(f"Fail count: {fail_count}")
    # output results with pd
    df_felix = pd.DataFrame(
        {"formal": df_test.formal, "informal": df_test.informal, "formal_pred": results}
    )
    df_felix.to_csv(out_csv_path, index=False)
