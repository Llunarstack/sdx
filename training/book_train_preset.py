"""Apply book/comic training presets to ``train.py`` argparse namespace."""

from __future__ import annotations

from typing import Any


def apply_book_train_preset_to_args(args: Any) -> None:
    """
    When ``--book-train-preset`` is set, fill unset train flags from book presets.

    Book training uses the **same** ``train.py`` / DiT-*-Text stack as general image gen;
    only defaults (AR blocks, caption hierarchy, guidance packs) differ.
    """
    preset_name = str(getattr(args, "book_train_preset", "") or "").strip().lower()
    if not preset_name:
        return

    from pipelines.book_comic.book_training_helpers import resolve_book_train_settings

    book_ar = str(getattr(args, "book_ar_profile", "") or "").strip()
    if book_ar and not str(getattr(args, "ar_profile", "") or "").strip():
        args.ar_profile = book_ar

    settings = resolve_book_train_settings(args)

    args.model = settings.model
    args.image_size = settings.image_size
    args.global_batch_size = settings.global_batch_size
    args.lr = settings.lr
    args.passes = settings.passes
    if settings.max_steps > 0:
        args.max_steps = settings.max_steps

    args.train_shortcomings_mitigation = settings.train_shortcomings_mitigation
    args.train_shortcomings_2d = settings.train_shortcomings_2d
    args.train_art_guidance_mode = settings.train_art_guidance_mode
    args.train_anatomy_guidance = settings.train_anatomy_guidance
    args.train_style_guidance_mode = settings.train_style_guidance_mode
    args.region_caption_mode = settings.region_caption_mode
    args.num_ar_blocks = settings.num_ar_blocks
    args.ar_block_order = settings.ar_block_order
    args.use_hierarchical_captions = settings.use_hierarchical_captions
    args.boost_adherence_caption = settings.boost_adherence_caption
