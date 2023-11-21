from transformers import AutoTokenizer
from neo_stif.components.utils import create_pointer_labels, create_label_map
from neo_stif.components.trying import PointingConverter

def tokenize_function_src_tgt(examples, tokenizer, src="informal", tgt="formal"):
    returned_dict = {f"{src}_{i}": j for i, j in tokenizer(examples[src]).items()}
    returned_dict.update({f"{tgt}_{i}": j for i, j in tokenizer(examples[tgt]).items()})
    return returned_dict


def generate_tokenized(
    examples, tokenizer, label_dict, point_converter, src="informal", tgt="formal"
):
    src_tokenized = tokenizer.tokenize(examples[src], add_special_tokens=True)
    tgt_tokenized = tokenizer.tokenize(examples[tgt], add_special_tokens=True)
    points = point_converter.compute_points(src_tokenized, " ".join(tgt_tokenized))
    label = create_pointer_labels(points, label_dict)
    point_indexes = [t.point_index for t in points]
    # change them to torch tensors
    label = label
    point_indexes = point_indexes
    return {f"tag_labels": label, f"point_labels": point_indexes}


def prepare_data_tagging_and_pointer(data, tokenizer, label_dict):
    data_preprocessed = data.map(
        tokenize_function_src_tgt,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )
    point_converter = PointingConverter({}, False)
    data = data_preprocessed.map(
        generate_tokenized,
        batched=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label_dict": label_dict,
            "point_converter": point_converter,
        },
    )
    return data, label_dict


