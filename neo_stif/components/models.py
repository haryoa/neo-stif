from ctypes import pointer
from typing import Optional
from torch import nn
from transformers import BertModel
import torch
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers.models.bert.modeling_bert import BertLayer
import math


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query_state,
        key_state,
        value_state,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(query_state)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(key_state)
            mixed_value_layer = self.value(value_state)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class PointerNetwork(nn.Module):
    def __init__(self, pointer_config, previous_hidden_dim: int = 768) -> None:
        super().__init__()
        self.pointer_config = pointer_config
        self.bert = BertModel(pointer_config)
        self.nll_loss = NLLLoss()
        self.word_embeddings = nn.Embedding(
            pointer_config.vocab_size,
            pointer_config.hidden_size,
            padding_idx=pointer_config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            pointer_config.max_position_embeddings, pointer_config.hidden_size
        )
        self.position_embedding_type = getattr(
            pointer_config, "position_embedding_type", "absolute"
        )
        self.LayerNorm = nn.LayerNorm(
            pointer_config.hidden_size, eps=pointer_config.layer_norm_eps
        )
        self.dropout = nn.Dropout(pointer_config.hidden_dropout_prob)
        after_embed_size = previous_hidden_dim + pointer_config.hidden_size * 2
        self.linear_after_embed = nn.Linear(
            after_embed_size, after_embed_size
        )
        # initialize 2 bert layer 
        self.register_buffer(
            "position_ids", torch.arange(pointer_config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        
        # intialize another pointer_config!
        from copy import deepcopy
        pointer_config_new = deepcopy(pointer_config)
        pointer_config_new.hidden_size = previous_hidden_dim + pointer_config.hidden_size * 2
        self.last_attention = BertSelfAttention(pointer_config_new)

        self.bert_layer = nn.ModuleList(
            [BertLayer(pointer_config_new) for _ in range(pointer_config_new.num_hidden_layers)]
        )

        self.gelu = nn.GELU()

    def embedding_layer(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, 0 : seq_length + 0]

        input_embeds = self.word_embeddings(input_ids)
        # embeddings = input_embeds
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            # make the dimension of position_embeddings the same as input_embeds
            position_embeddings = position_embeddings.expand_as(input_embeds)
            # embeddings += position_embeddings

        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return input_embeds, position_embeddings

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        previous_bert_output,
        labels=None,
    ):
        input_embeds, position_embeddings = self.embedding_layer(input_ids)

        # concate previous bert output with the input_embeds and embeddings
        pointer_input = torch.cat(
            (previous_bert_output, input_embeds, position_embeddings), dim=-1
        )
        # go to linear layer + gelu
        pointer_input = self.linear_after_embed(pointer_input)
        pointer_input = self.gelu(pointer_input)

        # go to bert
        input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(
            attention_mask, input_shape
        )

        bert_output = pointer_input
        for i, layer_module in enumerate(self.bert_layer):
            bert_output = layer_module(bert_output, extended_attention_mask)[0]

        # print(bert_output[0].shape)
        _, last_attention = self.last_attention(
            bert_output, pointer_input, bert_output, extended_attention_mask, output_attentions=True
        )
        if labels is not None:
            log_softmax_out = torch.nn.LogSoftmax(dim=-1)(last_attention)
            loss = self.nll_loss(
                log_softmax_out.view(-1, log_softmax_out.shape[-1]),
                labels.view(-1),
            )
            return loss, last_attention
        return None, last_attention
