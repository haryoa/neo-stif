# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for running inference with a Felix model."""

from typing import Optional, Sequence, Set

import numpy as np
import scipy.special


def get_number_of_masks(label):
  """Convert a tag to the number of MASK tokens it represents."""

  if '|' not in label:
    return 0
  return int(label.split('|')[1])


def _normalize_logits(logits):
  numerator = logits
  denominator = scipy.special.logsumexp(logits)
  return numerator - denominator


def beam_search_single_tagging(
    predicted_points_logits,
    good_indexes,
    sep_indexes,
    beam_size,
    end_index = 128,
    max_length = 128):
  """Returns the most likely (according to a beam search) sequence of indexes.

  Args:
    predicted_points_logits: Matrix of logits (timesteps x timesteps). Each
      timestep has logits for every other timestep.
    good_indexes: A restricted set of indexes which the beam must use. As such
      the problem becomes find the most likely permutation of these indexes.
    sep_indexes: A set of indexes for the [SEP] token. This ensure the last
      token is a [SEP].
    beam_size: The size of the beam.
    end_index: The index of the last token (excluding padding)
    max_length: The maximum length of the generation.

  Returns:
    The most likely sequence of indexes.
  """
  # -1 is useful for np.argpartition which splits on smallest.
  predicted_points = -1 * _normalize_logits(predicted_points_logits)
  sequences = [[0]]
  scores = [0]
  finished_sequences = []
  finished_scores = []
  for _ in range(max_length):
    assert len(sequences) == len(scores)
    candidate_scores = []
    candidate_sequences_reconstructor = []
    for j, (sequence, score) in enumerate(zip(sequences, scores)):
      sequence_set = set(sequence)
      next_scores = predicted_points[sequence[-1]]
      for index in range(end_index + 1):
        # Can't predict the same index twice.
        if index in sequence_set:
          continue
        # You must produce a good index.
        if index not in good_indexes:
          continue
        # The last token must be a [SEP].
        if len(sequence) == len(good_indexes) - 1:
          if index not in sep_indexes:
            continue
        # If there is only one SEP don't predict it till the end.
        elif index in sep_indexes and len(sep_indexes) == 1:
          continue

        candidate_scores.append(score + next_scores[index])
        # Don't construct a sequence for every candidate as this is expensive.
        # Instead store a way to reconstruct the sequence.
        candidate_sequences_reconstructor.append((j, index))

    if not candidate_scores:
      break

    if beam_size < 1:
      break
    if beam_size >= len(candidate_scores):
      top_n_indexes = list(range(len(candidate_scores)))
    else:
      # Get the N most likely sequences. (A full sort is not needed).
      top_n_indexes = np.argpartition(candidate_scores, beam_size)[:beam_size]

    new_sequences = []
    new_scores = []

    for top_n_index in top_n_indexes:
      sequence_index, token_index = candidate_sequences_reconstructor[
          top_n_index]
      # Reconstruct the sequence.
      new_sequence = sequences[sequence_index] + [token_index]
      new_score = candidate_scores[top_n_index]

      # For every completed beam we reduce the beamsize by 1.
      if len(new_sequence) == len(good_indexes):
        finished_sequences.append(new_sequence)
        finished_scores.append(-1 * new_score / len(new_sequence))
        beam_size -= 1
      else:
        new_sequences.append(new_sequence)
        new_scores.append(new_score)

    sequences = new_sequences
    scores = new_scores
    if beam_size < 1:
      break
  if not finished_sequences:
    return None

  return finished_sequences[np.argmax(finished_scores)]

def realize_beam_search(
    source_token_ids,
    ordered_source_indexes,
    tags,
    source_length,
    inverse_label_map,
    tokenizer,
):
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
    source_token_ids_set = set(ordered_source_indexes)
    out_tokens = []
    out_tokens_with_deletes = []
    for j, index in enumerate(ordered_source_indexes):
        token = tokenizer.convert_ids_to_tokens([source_token_ids[index]])
        out_tokens += token
        tag = inverse_label_map[tags[index]]
        out_tokens_with_deletes += token
        # Add the predicted MASK tokens.
        number_of_masks = get_number_of_masks(tag)
        # Can not add phrases after last token.
        if j == len(ordered_source_indexes) - 1:
            number_of_masks = 0
        masks = ["[MASK]"] * number_of_masks
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
            deleted_tokens = (
                ["[UNK]"]
                + list(
                    tokenizer.convert_ids_to_tokens(deleted_tokens)
                )
                + ["[PAD]"]
            )
            out_tokens_with_deletes += deleted_tokens
    return out_tokens_with_deletes

