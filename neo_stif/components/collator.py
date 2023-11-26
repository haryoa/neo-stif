# collator


from copy import deepcopy
import torch


class FelixCollator:
    def __init__(
        self,
        tokenizer,
        pad_label=-100,
        pad_label_as_input=80,
        src="informal",
        tgt="formal",
    ):
        # change the pad_label_as_input and point_label_as_input
        # these must be the len of the label vocab and point_label vocab
        self.tokenizer = tokenizer
        self.pad_label = pad_label
        self.src = src
        self.tgt = tgt
        self.pad_label_as_input = pad_label_as_input

    def __call__(self, batch):
        # batch is a list of dicts
        output_dict = {}
        informal_input_ids, informal_attention_mask = [
            [i[col] for i in batch]
            for col in [
                f"{self.src}_input_ids",
                f"{self.src}_attention_mask",
            ]
        ]
        formal_input_ids = [i[f"{self.tgt}_input_ids"] for i in batch]

        tag_label = [i["tag_labels"] for i in batch]

        tokenized_output = self.tokenizer.pad(
            {
                "input_ids": informal_input_ids,
                "attention_mask": informal_attention_mask,
            },
            return_tensors="pt",
        )
        tokenized_output["token_type_ids"] = torch.zeros_like(
            tokenized_output["input_ids"]
        )

        output_label = self.tokenizer.pad(
            {
                "input_ids": formal_input_ids,
            },
            return_tensors="pt",
        )
        out_label = output_label["input_ids"]
        # change pad token to -100
        out_label[out_label == self.tokenizer.pad_token_id] = self.pad_label
        # print(out_label)
        output_dict.update(tokenized_output)
        output_dict["labels"] = out_label
        # print(output_dict)
        # add tag_label to output_dict
        # each tag_label is a list of labels (list) with different length
        # pad first
        max_len = max([len(i) for i in tag_label])
        tag_label = [i + [self.pad_label] * (max_len - len(i)) for i in tag_label]
        tag_label = torch.tensor(tag_label)

        output_dict["tag_labels"] = tag_label
        output_dict["tag_labels_input"] = deepcopy(tag_label)
        output_dict["tag_labels_input"][
            output_dict["tag_labels_input"] == self.pad_label
        ] = self.pad_label_as_input

        # add point_label to output_dict
        # same as above
        point_label = [i["point_labels"] for i in batch]
        max_len = max([len(i) for i in point_label])
        point_label = [i + [self.pad_label] * (max_len - len(i)) for i in point_label]
        point_label = torch.tensor(point_label)

        output_dict["point_labels"] = point_label

        return output_dict


class FelixInsertionCollator:
    def __init__(self, tokenizer, mask_token_id=2, mask_label_id=-100):
        # change the pad_label_as_input and point_label_as_input
        # these must be the len of the label vocab and point_label vocab
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.vocab["[MASK]"]
        self.mask_label_id = mask_label_id

    def __call__(self, batch):
        # batch is a list of dicts
        output_dict = {}
        # important one is input_ids, attention_mask, token_type_ids and labels
        # input_ids is the informal sentence
        # labels is the formal sentence

        input_ids, attention_mask, token_type_ids, labels = [
            [i[col] for i in batch]
            for col in [
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "labels",
            ]
        ]

        # we pad like above
        tokenized_output = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
            return_tensors="pt",
        )

        # we pad the labels
        output_label = self.tokenizer.pad(
            {
                "input_ids": labels,
            },
            return_tensors="pt",
        )

        # now since we want to do MLM, our input is already masked
        # so, we need to change the labels to -100 for the masked tokens
        # and the rest to the actual label
        out_modified = output_label["input_ids"]

        # we need to find the mask token id
        # and then change the labels to -100

        masked_bool = tokenized_output["input_ids"] == self.mask_token_id
        # Now, change label if not masked
        out_modified[~masked_bool] = self.mask_label_id  # -100

        # collect them to output_dict
        output_dict.update(tokenized_output)
        output_dict["labels"] = out_modified

        return output_dict
