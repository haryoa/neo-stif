from torch import nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention
import torch
from torch.nn import CrossEntropyLoss, NLLLoss


class PointerNetwork(nn.Module):
    def __init__(self, pointer_config) -> None:
        super().__init__()
        self.pointer_config = pointer_config
        self.bert = BertModel(pointer_config)
        self.last_attention = BertSelfAttention(pointer_config)
        self.nll_loss = NLLLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)

        input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(
            attention_mask, input_shape
        )
        # print(bert_output[0].shape)
        _, last_attention = self.last_attention(
            bert_output[0], extended_attention_mask, output_attentions=True
        )
        if labels is not None:
            log_softmax_out = torch.nn.LogSoftmax(dim=-1)(last_attention)
            loss = self.nll_loss(
                log_softmax_out.view(-1, log_softmax_out.shape[-1]),
                labels.view(-1),
            )
            return loss, last_attention
        return None, last_attention
