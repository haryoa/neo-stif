def create_label_map(max_mask: int = 5, use_pointing: bool = True):
    """Creates label map for insertion model.

    Args:
        max_mask: Maximum number of masks.
        use_pointing: Whether to use pointing or not.

    Returns:
        label_map: A dictionary mapping label strings to label IDs.
    """
    label_map = {"PAD": 0, "SWAP": 1, "KEEP": 2, "DELETE": 3}
    # Create Insert 1 MASK to insertion N MASKS.
    for i in range(1, max_mask + 1):
        label_map[f"KEEP|{i}"] = len(label_map)
    if not use_pointing:
        label_map[f"DELETE|{i}"] = len(label_map)
    return label_map


def create_pointer_labels(points, label_map):
    """
    Creates labels for the pointing model.

    Args:
        points: List of pointing.Point objects.
        label_map: A dictionary mapping label strings to label IDs. 
    
    Returns:
        labels: List of label IDs, which correspond to a list of labels (KEEP,
        DELETE, MASK|1, MASK|2...).
    """
    labels = [t.added_phrase for t in points]
    point_indexes = [t.point_index for t in points]
    point_indexes_set = set(point_indexes)
    new_labels = []
    for i, added_phrase in enumerate(labels):
        if i not in point_indexes_set:
            new_labels.append(label_map["DELETE"])
        elif not added_phrase:
            new_labels.append(label_map["KEEP"])
        else:
            new_labels.append(label_map["KEEP|" + str(len(added_phrase.split()))])
    return new_labels


def get_pointer_and_label(x, label_dict, point_converter, tokenizer_bert):
    """
    Creates labels for the pointing model.

    Args:
        x: A tuple of informal and formal sentences.
        label_dict: A dictionary mapping label strings to label IDs.
        point_converter: A PointingConverter object.
        tokenizer_bert: A BertTokenizer object.
    
    Returns:
        point_indexes: List of next tokens (see pointing converter for more
        details) (ordered by source tokens).
        label: List of label IDs, which correspond to a list of labels (KEEP,
        DELETE, MASK|1, MASK|2...).
    """

    informal_instance = tokenizer_bert.tokenize(x.informal, add_special_tokens=True)
    formal_instance = tokenizer_bert.tokenize(x.formal, add_special_tokens=True)
    points = point_converter.compute_points(
        informal_instance, " ".join(formal_instance)
    )
    label = create_pointer_labels(points, label_dict)
    point_indexes = [t.point_index for t in points]
    return point_indexes, label
