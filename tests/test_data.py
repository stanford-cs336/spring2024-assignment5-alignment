#!/usr/bin/env python3
import logging
import math

import torch
from transformers import AutoTokenizer

from .adapters import get_packed_sft_dataset, run_iterate_batches
from .common import FIXTURES_PATH

logger = logging.getLogger(__name__)


def test_packed_sft_dataset():
    sft_sample_path = FIXTURES_PATH / "sft_sample.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("/data/Meta-Llama-3-8B")
    seq_length = 32
    packed_sft_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=sft_sample_path,
        seq_length=seq_length,
        shuffle=False,
    )
    assert len(packed_sft_dataset) == 75
    examples_with_bos = []
    examples_with_eos = []
    for example_idx, example in enumerate(packed_sft_dataset):
        assert "input_ids" in example
        assert "labels" in example

        assert len(example["input_ids"]) == seq_length
        assert len(example["labels"]) == seq_length
        # Make sure the labels are offset by 1 from the input_ids
        assert torch.allclose(example["input_ids"][1:], example["labels"][:-1])

        assert (
            example["input_ids"].dtype == torch.long
            or example["input_ids"].dtype == torch.int64
        )
        assert (
            example["labels"].dtype == torch.long
            or example["labels"].dtype == torch.int64
        )

        decoded_text = tokenizer.decode(example["input_ids"])
        if tokenizer.bos_token in decoded_text:
            examples_with_bos.append(example_idx)
        if tokenizer.eos_token in decoded_text:
            examples_with_eos.append(example_idx)

    # There should be exactly 5 examples with the bos token
    assert len(examples_with_bos) == 5
    # There should be exactly 4 examples with the eos token,
    # since the last example gets dropped
    assert len(examples_with_eos) == 4

    # Check that the examples with the bos / eos are the ones that we expect
    assert examples_with_bos == [0, 28, 43, 50, 67]
    assert examples_with_eos == [28, 43, 50, 67]

    # Check that shuffling works by ensuring that it returns different data
    # than the unshuffled dataset. Note that there's a small chance that it
    # just happens that the shuffling preserves the original order.
    shuffled_packed_sft_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=sft_sample_path,
        seq_length=seq_length,
        shuffle=True,
    )
    all_unshuffled_examples = []
    all_shuffled_examples = []
    for example, shuffled_example in zip(
        packed_sft_dataset, shuffled_packed_sft_dataset
    ):
        all_unshuffled_examples.append({k: v.tolist() for k, v in example.items()})
        all_shuffled_examples.append(
            {k: v.tolist() for k, v in shuffled_example.items()}
        )
    assert all_unshuffled_examples != all_shuffled_examples


def test_iterate_batches():
    sft_sample_path = FIXTURES_PATH / "sft_sample.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("/data/Meta-Llama-3-8B")
    seq_length = 32
    batch_size = 8
    packed_sft_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=sft_sample_path,
        seq_length=seq_length,
        shuffle=True,
    )
    train_dataloader = run_iterate_batches(
        dataset=packed_sft_dataset, batch_size=batch_size, shuffle=True
    )
    assert len(train_dataloader) == math.ceil(75 / batch_size)
    for batch_idx, batch in enumerate(train_dataloader):
        # Make sure each of input_ids and labels is a (batch_size, seq_length) tensor, except
        # for the last batch (which can be less than batch_size items)
        if batch_idx != len(train_dataloader) - 1:
            assert batch["input_ids"].shape == (batch_size, seq_length)
            assert batch["labels"].shape == (batch_size, seq_length)

        assert (
            batch["input_ids"].dtype == torch.long
            or batch["input_ids"].dtype == torch.int64
        )
        assert (
            batch["labels"].dtype == torch.long or batch["labels"].dtype == torch.int64
        )
