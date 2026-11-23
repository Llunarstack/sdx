"""``python -m scripts.tools ui``: argv building, validation, file serving and the one-at-a-time runner."""

from __future__ import annotations

import http.client
import json
import sys
import threading
import time

import pytest
from scripts.tools import ui

# Stand-in for sample.py: prints progress and writes the --out image.
FAKE = (
    "import sys;"
    "out=[a.split('=',1)[1] for a in sys.argv if a.startswith('--out=')][0];"
    "print('step 1/2');print('step 2/2');open(out,'wb').write(b'png')"
)


@pytest.fixture
def ckpt(tmp_path):
    p = tmp_path / "best.pt"
    p.write_bytes(b"x")
    return str(p)


def test_build_argv_from_form(tmp_path, ckpt):
    form = {
        "ckpt": ckpt,
        "prompt": "-a cat",
        "negative": "blurry",
        "width": "512",
        "steps": "20",
        "cfg": "4.5",
        "seed": "7",
        "num": "2",
        "detail_amount": "0.2",
        "extra": "--solver unipc --style 'oil paint'",
        "height": "",
    }
    argv, settings = ui.build_argv(form, tmp_path, "run1", ["py", "sample.py"])
    assert argv[:2] == ["py", "sample.py"]
    assert "--prompt=-a cat" in argv and "--negative-prompt=blurry" in argv
    assert f"--out={tmp_path / 'run1.png'}" in argv
    for pair in (["--seed", "7"], ["--width", "512"], ["--cfg-scale", "4.5"], ["--detail-amount", "0.2"]):
        i = argv.index(pair[0])
        assert argv[i : i + 2] == pair
    assert "--height" not in argv
    assert argv[-4:] == ["--solver", "unipc", "--style", "oil paint"]
    assert settings["seed"] == 7 and settings["num"] == 2 and settings["extra"]


def test_random_seed_and_preset(tmp_path, ckpt):
    argv, settings = ui.build_argv({"ckpt": ckpt, "prompt": "x"}, tmp_path, "r", ["py"])
    assert 0 <= settings["seed"] < 2**31
    names = ui.preset_names()
    if names:
        argv, _ = ui.build_argv({"ckpt": ckpt, "prompt": "x", "preset": names[0]}, tmp_path, "r", ["py"])
        assert ["--preset", names[0]] == argv[argv.index("--preset") : argv.index("--preset") + 2]


@pytest.mark.parametrize(
    "bad",
    [
        {"ckpt": "missing.pt"},
        {"prompt": ""},
        {"steps": "0"},
        {"steps": "abc"},
        {"cfg": "nan"},
        {"width": "99999"},
        {"num": "17"},
        {"seed": "-5"},
        {"detail_amount": "2"},
        {"preset": "no-such-preset"},
        {"extra": "--out=evil.png"},
        {"extra": "--ckpt other.pt"},
        {"extra": "--jobs-jsonl jobs.jsonl"},
        {"extra": "--jobs-jsonl=jobs.jsonl"},
        {"extra": "--jobs-json jobs.jsonl"},
        {"extra": "'unclosed"},
        {"prompt": "x" * 5000},
    ],
)
def test_validation_rejects(tmp_path, ckpt, bad):
    with pytest.raises(ValueError):
        ui.build_argv({"ckpt": ckpt, "prompt": "ok", **bad}, tmp_path, "r", ["py"])


@pytest.mark.parametrize(
    "flag",
    [
        "--save-attn",
        "--pick-report-json",
        "--feedback-log",
        "--log-feedback",
        "--save-character-session",
        "--export-comfy-workflow",
        "--face-restore-shell",
    ],
)
def test_extra_rejects_external_paths_and_commands(tmp_path, ckpt, flag):
    with pytest.raises(ValueError):
        ui.build_argv({"ckpt": ckpt, "prompt": "ok", "extra": f"{flag}=outside"}, tmp_path, "r", ["py"])


def test_safe_file_rejects_traversal(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.png").write_bytes(b"1")
    (out / "notes.txt").write_text("t")
    (tmp_path / "secret.png").write_bytes(b"s")
    assert ui.safe_file(out, "a.png") == (out / "a.png").resolve()
    for name in ("../secret.png", "..%2fsecret.png", "..\\secret.png", "C:secret.png", ".hidden.png", "notes.txt", ""):
        assert ui.safe_file(out, name) is None


@pytest.fixture
def server(tmp_path):
    runner = ui.Runner(tmp_path / "out", [sys.executable, "-c", FAKE])
    srv = ui.ThreadingHTTPServer(("127.0.0.1", 0), ui.make_handler(runner))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, runner
    srv.shutdown()
    srv.server_close()


def _req(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, data


def _post(srv, form, **headers):
    return _req(srv, "POST", "/api/generate", json.dumps(form), {"Content-Type": "application/json", **headers})


def test_server_generate_history_and_files(server, ckpt):
    srv, runner = server
    status, page = _req(srv, "GET", "/")
    assert status == 200 and b"SDX Sampler" in page and b"__PRESETS__" not in page
    status, data = _post(srv, {"ckpt": ckpt, "prompt": "a cat", "seed": "3"})
    assert status == 202
    run_id = json.loads(data)["run_id"]
    deadline = time.time() + 20
    while runner.state(0)["running"] and time.time() < deadline:
        time.sleep(0.05)
    st = json.loads(_req(srv, "GET", "/api/state?since=0")[1])
    assert st["returncode"] == 0 and st["lines"] == ["step 1/2", "step 2/2"] and st["next"] == 2
    hist = json.loads(_req(srv, "GET", "/api/history")[1])
    assert hist[0]["run_id"] == run_id and hist[0]["settings"]["seed"] == 3
    assert hist[0]["images"] == [f"{run_id}.png"]
    assert _req(srv, "GET", f"/files/{run_id}.png") == (200, b"png")
    assert _req(srv, "GET", "/files/..%2f..%2fbest.pt")[0] == 404
    assert _req(srv, "GET", f"/files/{run_id}.json")[0] == 404


def test_server_rejects_busy_bad_input_and_cross_site(server, ckpt):
    srv, runner = server
    assert _post(srv, {"ckpt": ckpt, "prompt": ""})[0] == 400
    assert _req(srv, "POST", "/api/generate", "{}", {"Content-Type": "text/plain"})[0] == 415
    assert _post(srv, {"ckpt": ckpt, "prompt": "x"}, Origin="https://evil.example")[0] == 403
    assert _req(srv, "GET", "/", headers={"Host": "evil.example"})[0] == 403
    runner.running = True
    assert _post(srv, {"ckpt": ckpt, "prompt": "x"})[0] == 409
    runner.running = False
