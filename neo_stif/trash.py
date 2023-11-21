## POINTER NETWORK INFERENCE

def _realize_beam_search(self, source_token_ids,
                           ordered_source_indexes,
                           tags,
                           source_length):
    """Returns realized prediction using indexes and tags.

    TODO: Refactor this function to share code with
    `_create_masked_source` from insertion_converter.py to reduce code
    duplication and to ensure that the insertion example creation is consistent
    between preprocessing and prediction.

    Args:
      source_token_ids: List of source token ids.
      ordered_source_indexes: The order in which the kept tokens should be
        realized.
      tags: a List of tags.
      source_length: How long is the source input (excluding padding).

    Returns:
      Realized predictions (with deleted tokens).
    """
    # Need to help type checker.
    self._inverse_label_map = cast(Mapping[int, str], self._inverse_label_map)

    source_token_ids_set = set(ordered_source_indexes)
    out_tokens = []
    out_tokens_with_deletes = []
    for j, index in enumerate(ordered_source_indexes):
      token = self._builder.tokenizer.convert_ids_to_tokens(
          [source_token_ids[index]])
      out_tokens += token
      tag = self._inverse_label_map[tags[index]]
      if self._use_open_vocab:
        out_tokens_with_deletes += token
        # Add the predicted MASK tokens.
        number_of_masks = insertion_converter.get_number_of_masks(tag)
        # Can not add phrases after last token.
        if j == len(ordered_source_indexes) - 1:
          number_of_masks = 0
        masks = [constants.MASK] * number_of_masks
        out_tokens += masks
        out_tokens_with_deletes += masks

        # Find the deleted tokens, which appear after the current token.
        deleted_tokens = []
        for i in range(index + 1, source_length):
          if i in source_token_ids_set:
            break
          deleted_tokens.append(source_token_ids[i])
        # Bracket the deleted tokens, between unused0 and unused1.
        if deleted_tokens:
          deleted_tokens = [constants.DELETE_SPAN_START] + list(
              self._builder.tokenizer.convert_ids_to_tokens(deleted_tokens)) + [
                  constants.DELETE_SPAN_END
              ]
          out_tokens_with_deletes += deleted_tokens
      # Add the predicted phrase.
      elif '|' in tag:
        pos_pipe = tag.index('|')
        added_phrase = tag[pos_pipe + 1:]
        out_tokens.append(added_phrase)

    if not self._use_open_vocab:
      out_tokens_with_deletes = out_tokens
    assert (
        out_tokens_with_deletes[0] == (constants.CLS)
    ), (f' {out_tokens_with_deletes} did not start/end with the correct tokens '
        f'{constants.CLS}, {constants.SEP}')
    return out_tokens_with_deletes


tag_embedding = self._tag_embedding_layer(edit_tags)
position_embedding = self._position_embedding_layer(tag_embedding)
edit_tagged_sequence_output = self._edit_tagged_sequence_output_layer(
    tf.keras.layers.concatenate(
        [bert_output, tag_embedding, position_embedding]))

intermediate_query_embeddings = edit_tagged_sequence_output
if self._bert_config.query_transformer:
    attention_mask = self._self_attention_mask_layer(
        intermediate_query_embeddings, input_mask)
    for _ in range(int(self._bert_config.query_transformer)):
    intermediate_query_embeddings = self._transformer_query_layer(
        [intermediate_query_embeddings, attention_mask])

query_embeddings = self._query_embeddings_layer(
    intermediate_query_embeddings)

key_embeddings = self._key_embeddings_layer(edit_tagged_sequence_output)

pointing_logits = self._attention_scores(query_embeddings, key_embeddings,
                                            tf.cast(input_mask, tf.float32))

