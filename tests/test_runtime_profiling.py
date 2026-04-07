from utils.runtime.profiling import ProfileConfig, consume_profile_args, run_with_cprofile


def test_consume_profile_args_strips_flags():
    argv = ["prog", "--profile-out", "x.prof", "--profile-sort", "tottime", "--profile-top", "10", "--ckpt", "m.pt"]
    rest, cfg = consume_profile_args(argv)
    assert rest == ["prog", "--ckpt", "m.pt"]
    assert cfg is not None
    assert cfg.out_path == "x.prof"
    assert cfg.sort_key == "tottime"
    assert cfg.top_n == 10


def test_consume_profile_args_no_profile():
    argv = ["prog", "--ckpt", "m.pt"]
    rest, cfg = consume_profile_args(argv)
    assert rest == argv
    assert cfg is None


def test_run_with_cprofile_writes_files(tmp_path):
    out = tmp_path / "run.prof"
    cfg = ProfileConfig(out_path=str(out), sort_key="cumulative", top_n=5)

    def _work():
        x = 0
        for i in range(1000):
            x += i

    run_with_cprofile(_work, cfg)
    assert out.is_file()
    summary = tmp_path / "run.prof.txt"
    assert summary.is_file()
    assert summary.stat().st_size > 0
