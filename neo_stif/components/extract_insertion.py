from neo_stif.components.insert_convert import get_number_of_masks


def create_masked_source(source_tokens, labels, source_indexes,
                        target_tokens, label_map):
    """Realizes source_tokens & adds deleted to source_tokens and target_tokens.

    Args:
        source_tokens: List of source tokens.
        labels: List of label IDs, which correspond to a list of labels (KEEP,
        DELETE, MASK|1, MASK|2...).
        source_indexes: List of next tokens (see pointing converter for more
        details) (ordered by source tokens)
        target_tokens: Optional list of target tokens. Only provided when
        constructing training examples.

    Returns:
        masked_tokens: The source input for the insertion model, including MASK
        tokens and bracketed deleted tokens.
        target_tokens: The target tokens for the insertion model, where mask
        tokens are replaced with the actual token, also includes bracketed
        deleted tokens.
    """
    # well its unused...
    DELETE_SPAN_START = '[UNK]'
    DELETE_SPAN_END = '[PAD]'
    current_index = 0
    masked_tokens = []

    label_map_inverse = {v: k for k, v in label_map.items()}

    kept_tokens = set([0])
    for _ in range(len(source_tokens)):
        current_index = source_indexes[current_index]
        kept_tokens.add(current_index)
        # Token is deleted.
        if current_index == 0:
            break

    current_index = 0
    for _ in range(len(source_tokens)):
        source_token = source_tokens[current_index]
        deleted_tokens = []
        # Looking forward finding all deleted tokens.
        for i in range(current_index + 1, len(source_tokens)):
        ## If not a deleted token.
            if i in kept_tokens:
                break
            deleted_tokens.append(source_tokens[i])

        # Add deleted tokens to masked_tokens and target_tokens.
        masked_tokens.append(source_token)
        # number_of_masks specifies the number MASKED tokens which
        # are added to masked_tokens.
        number_of_masks = get_number_of_masks(
            label_map_inverse[labels[current_index]])
        for _ in range(number_of_masks):
            masked_tokens.append('[MASK]')
        if deleted_tokens:
            masked_tokens_length = len(masked_tokens)
            bracketed_deleted_tokens = ([DELETE_SPAN_START] +
                                    deleted_tokens +
                                        [DELETE_SPAN_END])
            target_tokens = (
                target_tokens[:masked_tokens_length] + bracketed_deleted_tokens +
                target_tokens[masked_tokens_length:])
            masked_tokens += bracketed_deleted_tokens

        current_index = source_indexes[current_index]
        if current_index == 0:
            break
    return masked_tokens, target_tokens
