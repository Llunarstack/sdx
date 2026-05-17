"""Book training preset wiring for ``train.py``."""

from __future__ import annotations

import argparse

from training.book_train_preset import apply_book_train_preset_to_args


def test_apply_book_train_preset_sets_model() -> None:
    args = argparse.Namespace(
        book_train_preset="fast",
        book_ar_profile="auto",
        model="",
        image_size=0,
        global_batch_size=0,
        lr=0.0,
        passes=-1,
        max_steps=-1,
        ar_profile="",
        num_ar_blocks=-1,
        ar_block_order="",
    )
    apply_book_train_preset_to_args(args)
    assert args.model == "DiT-B/2-Text"
    assert args.use_hierarchical_captions is True
    assert args.boost_adherence_caption is True


def test_apply_book_train_preset_off_is_noop() -> None:
    args = argparse.Namespace(book_train_preset="", model="DiT-XL/2-Text")
    apply_book_train_preset_to_args(args)
    assert args.model == "DiT-XL/2-Text"
