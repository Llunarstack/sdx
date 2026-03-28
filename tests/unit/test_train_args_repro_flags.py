"""Train CLI/config mapping for reproducibility and strict warning flags."""

from training.train_args import build_train_config_from_args
from training.train_cli_parser import build_train_arg_parser


def test_train_args_repro_flags_default_and_override():
    parser = build_train_arg_parser()

    args_default = parser.parse_args([])
    cfg_default = build_train_config_from_args(args_default)
    assert cfg_default.save_run_manifest is True
    assert cfg_default.strict_warnings is False

    args_override = parser.parse_args(["--no-save-run-manifest", "--strict-warnings"])
    cfg_override = build_train_config_from_args(args_override)
    assert cfg_override.save_run_manifest is False
    assert cfg_override.strict_warnings is True