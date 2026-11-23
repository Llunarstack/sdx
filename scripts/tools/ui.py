"""Local web UI for ``sample.py`` (stdlib only, bound to 127.0.0.1)::

    python -m scripts.tools ui [--port 7860] [--out-dir outputs/ui]

One generation at a time; its log streams to the page. Results and the history strip come from the output
folder (one ``<run>.json`` settings record per run, written next to the images).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO = Path(__file__).resolve().parents[2]
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
LOCAL_HOSTS = ("127.0.0.1", "localhost")
MAX_LINES = 400
MAX_BODY = 64 * 1024
# form field -> (type, min, max, sample.py flag); a blank field keeps the sample.py / preset default.
NUMERIC = {
    "width": (int, 64, 4096, "--width"),
    "height": (int, 64, 4096, "--height"),
    "steps": (int, 1, 200, "--steps"),
    "cfg": (float, 0.0, 30.0, "--cfg-scale"),
    "num": (int, 1, 16, "--num"),
    "detail_amount": (float, -1.0, 1.0, "--detail-amount"),
}
# The UI owns these: history and file serving depend on them.
RESERVED_FLAGS = (
    "--out",
    "--ckpt",
    "--jobs-jsonl",
    "--save-attn",
    "--pick-report-json",
    "--feedback-log",
    "--log-feedback",
    "--save-character-session",
    "--export-comfy-workflow",
    "--face-restore-shell",
)


def preset_names() -> list[str]:
    try:
        from config.presets import PRESETS
    except Exception:
        return []
    return sorted(PRESETS)


def _number(key: str, raw: str, typ: type, lo: float, hi: float) -> int | float:
    try:
        v = typ(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number") from None
    if not (math.isfinite(v) and lo <= v <= hi):
        raise ValueError(f"{key} must be between {lo} and {hi}")
    return v


def build_argv(form: dict, out_dir: Path, run_id: str, command: list[str]) -> tuple[list[str], dict]:
    """Validated ``sample.py`` argv for a form submission plus the settings recorded in history."""

    def text(key: str, limit: int) -> str:
        v = str(form.get(key) or "").strip()
        if len(v) > limit:
            raise ValueError(f"{key} is longer than {limit} characters")
        return v

    ckpt, prompt, negative = text("ckpt", 1024), text("prompt", 4000), text("negative", 4000)
    if not ckpt or not Path(ckpt).expanduser().is_file():
        raise ValueError(f"checkpoint not found: {ckpt or '(empty)'}")
    if not prompt:
        raise ValueError("prompt is empty")
    settings: dict = {"ckpt": ckpt, "prompt": prompt, "negative": negative}
    argv = [
        *command,
        f"--ckpt={Path(ckpt).expanduser().resolve()}",
        f"--prompt={prompt}",
        f"--out={out_dir / f'{run_id}.png'}",
        "--save-prompt",
    ]
    if negative:
        argv.append(f"--negative-prompt={negative}")
    preset = text("preset", 64)
    if preset:
        if preset not in preset_names():
            raise ValueError(f"unknown preset: {preset}")
        argv += ["--preset", preset]
        settings["preset"] = preset
    seed_raw = text("seed", 20)
    seed = secrets.randbelow(2**31) if seed_raw in ("", "-1") else _number("seed", seed_raw, int, 0, 2**32 - 1)
    argv += ["--seed", str(seed)]
    settings["seed"] = seed
    for key, (typ, lo, hi, flag) in NUMERIC.items():
        raw = text(key, 32)
        if raw:
            settings[key] = v = _number(key, raw, typ, lo, hi)
            argv += [flag, str(v)]
    extra = text("extra", 2000)
    try:
        extra_args = shlex.split(extra)
    except ValueError as e:
        raise ValueError(f"extra args: {e}") from None
    for a in extra_args:
        name = a.split("=", 1)[0]
        if len(name) > 2 and any(f.startswith(name) for f in RESERVED_FLAGS):
            raise ValueError(f"extra args: {name} is set by the UI")
    if extra_args:
        argv += extra_args
        settings["extra"] = extra
    return argv, settings


def safe_file(out_dir: Path, name: str) -> Path | None:
    """An image directly inside ``out_dir`` named ``name``, else None (no traversal, drives or dotfiles)."""
    name = unquote(name)
    if not name or name.startswith(".") or any(c in name for c in "/\\:\0"):
        return None
    root = out_dir.resolve()
    p = (root / name).resolve()
    if p.parent != root or p.suffix.lower() not in IMAGE_TYPES or not p.is_file():
        return None
    return p


def images_for(out_dir: Path, run_id: str) -> list[str]:
    return sorted(p.name for p in out_dir.glob(f"{run_id}*") if p.suffix.lower() in IMAGE_TYPES)


def history(out_dir: Path, limit: int = 40) -> list[dict]:
    """Past runs (newest first) from the ``<run>.json`` records the UI writes."""
    out = []
    for p in sorted(out_dir.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("run_id") == p.stem:
            rec["images"] = images_for(out_dir, p.stem)
            out.append(rec)
            if len(out) >= limit:
                break
    return out


class Runner:
    """Runs one ``sample.py`` subprocess at a time and keeps its last ``MAX_LINES`` output lines."""

    def __init__(self, out_dir: Path, command: list[str]):
        self.out_dir, self.command = out_dir, command
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.running, self.run_id, self.returncode = False, "", None
        self.lines: list[str] = []
        self.base = 0  # index of lines[0] in the run's full output

    def start(self, form: dict) -> str:
        with self.lock:
            if self.running:
                raise RuntimeError("a generation is already running; wait for it to finish")
            run_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
            argv, settings = build_argv(form, self.out_dir, run_id, self.command)
            self.running, self.run_id, self.returncode, self.lines, self.base = True, run_id, None, [], 0
        threading.Thread(target=self._run, args=(run_id, argv, settings), daemon=True).start()
        return run_id

    def _add(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)
            if len(self.lines) > MAX_LINES:
                drop = len(self.lines) - MAX_LINES
                del self.lines[:drop]
                self.base += drop

    def _run(self, run_id: str, argv: list[str], settings: dict) -> None:
        rec = {"run_id": run_id, "created": time.time(), "settings": settings}
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # Text mode turns tqdm's "\r" refreshes into separate lines.
            self.proc = subprocess.Popen(
                argv,
                cwd=REPO,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in self.proc.stdout:
                if line.strip():
                    self._add(line.rstrip())
            rc = self.proc.wait()
        except OSError as e:
            self._add(f"failed to start sample.py: {e}")
            rc = -1
        rec["returncode"] = rc
        try:
            (self.out_dir / f"{run_id}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        except OSError as e:
            self._add(f"could not write the run record: {e}")
        with self.lock:
            self.running, self.returncode, self.proc = False, rc, None

    def state(self, since: int) -> dict:
        with self.lock:
            start = max(0, since - self.base)
            return {
                "running": self.running,
                "run_id": self.run_id,
                "returncode": self.returncode,
                "lines": self.lines[start:],
                "next": self.base + len(self.lines),
            }

    def stop(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()


def make_handler(runner: Runner) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "sdx-ui"

        def log_message(self, *_args) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def _local(self) -> bool:
            # Host / Origin checks block DNS rebinding and cross-site POSTs from other pages in the browser.
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            origin = self.headers.get("Origin")
            return host in LOCAL_HOSTS and (not origin or urlparse(origin).hostname in LOCAL_HOSTS)

        def do_GET(self) -> None:
            if not self._local():
                return self._json(403, {"error": "local requests only"})
            url = urlparse(self.path)
            if url.path == "/":
                presets = json.dumps(preset_names()).replace("<", "\\u003c")
                return self._send(200, PAGE.replace("__PRESETS__", presets).encode("utf-8"), "text/html; charset=utf-8")
            if url.path == "/api/state":
                try:
                    since = int(parse_qs(url.query).get("since", ["0"])[0])
                except ValueError:
                    since = 0
                return self._json(200, runner.state(since))
            if url.path == "/api/history":
                return self._json(200, history(runner.out_dir))
            if url.path.startswith("/files/"):
                p = safe_file(runner.out_dir, url.path[len("/files/") :])
                if p is not None:
                    return self._send(200, p.read_bytes(), IMAGE_TYPES[p.suffix.lower()])
            return self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._local():
                return self._json(403, {"error": "local requests only"})
            if urlparse(self.path).path != "/api/generate":
                return self._json(404, {"error": "not found"})
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
                return self._json(415, {"error": "send JSON"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 < length <= MAX_BODY:
                return self._json(413, {"error": "bad request size"})
            try:
                form = json.loads(self.rfile.read(length))
            except ValueError:
                return self._json(400, {"error": "invalid JSON"})
            if not isinstance(form, dict):
                return self._json(400, {"error": "expected a JSON object"})
            try:
                run_id = runner.start(form)
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            except RuntimeError as e:
                return self._json(409, {"error": str(e)})
            return self._json(202, {"run_id": run_id})

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local web UI for sample.py (127.0.0.1 only).")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "ui"), help="Where images and run records go.")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(out_dir, [sys.executable, str(REPO / "sample.py")])
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runner))
    print(f"SDX UI on http://127.0.0.1:{server.server_port}/  (outputs: {out_dir}; Ctrl+C stops)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runner.stop()
    return 0


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDX Sampler</title>
<style>
:root {
  --bg: #f6f6f4; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --line: #deded8;
  --accent: #2f5bd3; --accent-text: #ffffff; --bad: #b3261e; --log: #f0f0ec;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141413; --panel: #1e1e1c; --text: #ecece8; --muted: #9b9b94; --line: #33332f;
    --accent: #7f9cff; --accent-text: #10131c; --bad: #ff8a80; --log: #191917;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.45 system-ui, sans-serif; }
header { padding: 14px 16px; border-bottom: 1px solid var(--line); font-weight: 600; }
main { display: grid; gap: 16px; padding: 16px; grid-template-columns: minmax(0, 1fr); max-width: 1400px; margin: 0 auto; }
@media (min-width: 900px) { main { grid-template-columns: 380px minmax(0, 1fr); } }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; min-width: 0; }
label { display: block; font-size: 13px; color: var(--muted); margin: 10px 0 4px; }
input, select, textarea { width: 100%; padding: 8px 10px; border: 1px solid var(--line); border-radius: 7px;
  background: var(--bg); color: var(--text); font: inherit; }
textarea { resize: vertical; min-height: 64px; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.seed { display: flex; gap: 6px; }
button { padding: 9px 14px; border: 1px solid var(--line); border-radius: 7px; background: var(--panel);
  color: var(--text); font: inherit; cursor: pointer; }
button.primary { width: 100%; margin-top: 14px; background: var(--accent); color: var(--accent-text); border: 0; font-weight: 600; }
button:disabled { opacity: .55; cursor: default; }
#status { margin: 0 0 8px; min-height: 1.4em; }
#status.bad { color: var(--bad); }
pre { background: var(--log); border: 1px solid var(--line); border-radius: 7px; padding: 8px; margin: 0;
  max-height: 180px; overflow: auto; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
pre:empty { display: none; }
#results { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); margin-top: 12px; }
#results img { width: 100%; border-radius: 7px; display: block; }
#settings { font-size: 13px; color: var(--muted); margin-top: 8px; word-break: break-word; }
h2 { font-size: 14px; margin: 18px 0 8px; }
#history { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 6px; }
#history button { flex: 0 0 auto; padding: 0; width: 96px; height: 96px; overflow: hidden; }
#history img { width: 100%; height: 100%; object-fit: cover; display: block; }
.hint { font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<header>SDX Sampler</header>
<main>
<section>
<form id="f" autocomplete="off">
  <label for="ckpt">Checkpoint path</label>
  <input id="ckpt" name="ckpt" required placeholder="results/run/best.pt">
  <label for="prompt">Prompt</label>
  <textarea id="prompt" name="prompt" required></textarea>
  <label for="negative">Negative prompt</label>
  <textarea id="negative" name="negative"></textarea>
  <label for="preset">Preset</label>
  <select id="preset" name="preset"><option value="">(none)</option></select>
  <div class="row">
    <div><label for="width">Width</label><input id="width" name="width" type="number" min="64" max="4096" step="8" placeholder="auto"></div>
    <div><label for="height">Height</label><input id="height" name="height" type="number" min="64" max="4096" step="8" placeholder="auto"></div>
    <div><label for="steps">Steps</label><input id="steps" name="steps" type="number" min="1" max="200" placeholder="default"></div>
    <div><label for="cfg">CFG</label><input id="cfg" name="cfg" type="number" min="0" max="30" step="0.1" placeholder="default"></div>
    <div><label for="seed">Seed</label><div class="seed"><input id="seed" name="seed" type="number" min="-1" placeholder="random">
      <button type="button" id="rand" aria-label="Random seed">Random</button></div></div>
    <div><label for="num">Images</label><input id="num" name="num" type="number" min="1" max="16" value="1"></div>
  </div>
  <label for="detail_amount">Detail amount</label>
  <input id="detail_amount" name="detail_amount" type="number" min="-1" max="1" step="0.05" placeholder="0 (off)">
  <div class="hint">0.1-0.3 adds fine detail; negative values smooth.</div>
  <label for="extra">Extra args</label>
  <input id="extra" name="extra" placeholder="--solver unipc --scheduler ays_dit">
  <div class="hint">Passed to sample.py as-is (shell-style quoting; use / or quotes for Windows paths).</div>
  <button class="primary" id="go" type="submit">Generate</button>
</form>
</section>
<section>
  <p id="status">Idle.</p>
  <pre id="log" aria-live="polite"></pre>
  <div id="results"></div>
  <div id="settings"></div>
  <h2>History</h2>
  <div id="history"></div>
</section>
</main>
<script>
const PRESETS = __PRESETS__;
const KEEP = ["ckpt", "prompt", "negative", "preset", "width", "height", "steps", "cfg", "num", "detail_amount", "extra"];
const $ = (id) => document.getElementById(id);
const form = $("f"), log = $("log");
let since = 0, runs = [];

for (const p of PRESETS) $("preset").append(new Option(p, p));
try {
  const saved = JSON.parse(localStorage.getItem("sdx-ui") || "{}");
  for (const k of KEEP) if (saved[k] != null && form.elements[k]) form.elements[k].value = saved[k];
} catch (e) {}

function status(msg, bad) { $("status").textContent = msg; $("status").className = bad ? "bad" : ""; }
const fileUrl = (name) => "/files/" + encodeURIComponent(name);

function show(run) {
  const box = $("results");
  box.replaceChildren();
  $("settings").textContent = "";
  if (!run) return;
  for (const name of run.images) {
    const a = document.createElement("a"), img = document.createElement("img");
    a.href = fileUrl(name); a.target = "_blank"; img.src = fileUrl(name); img.alt = run.settings.prompt;
    a.append(img); box.append(a);
  }
  const s = run.settings, parts = [];
  for (const [k, v] of Object.entries(s)) if (k !== "ckpt") parts.push(k + ": " + v);
  $("settings").textContent = parts.join(" | ") + (run.returncode ? " | exit " + run.returncode : "");
}

async function loadHistory() {
  runs = await (await fetch("/api/history")).json();
  const strip = $("history");
  strip.replaceChildren();
  for (const run of runs) {
    const b = document.createElement("button");
    b.type = "button"; b.title = run.settings.prompt + " (seed " + run.settings.seed + ")";
    if (run.images.length) {
      const img = document.createElement("img");
      img.src = fileUrl(run.images[0]); img.alt = ""; img.loading = "lazy"; b.append(img);
    } else b.textContent = run.returncode ? "failed" : "no image";
    b.onclick = () => show(run);
    strip.append(b);
  }
  return runs;
}

async function poll() {
  const st = await (await fetch("/api/state?since=" + since)).json();
  if (st.lines.length) {
    log.textContent = (log.textContent + st.lines.join("\\n") + "\\n").slice(-40000);
    log.scrollTop = log.scrollHeight;
  }
  since = st.next;
  if (st.running) {
    status("Generating... " + (st.lines.at(-1) || ""));
    $("go").disabled = true;
    setTimeout(poll, 700);
    return;
  }
  $("go").disabled = false;
  if (!st.run_id) return;
  status(st.returncode === 0 ? "Done." : "Failed (exit " + st.returncode + "). See the log.", st.returncode !== 0);
  await loadHistory();
  show(runs.find((r) => r.run_id === st.run_id));
}

$("rand").onclick = () => { $("seed").value = Math.floor(Math.random() * 2147483647); };

form.onsubmit = async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(form));
  try { localStorage.setItem("sdx-ui", JSON.stringify(Object.fromEntries(KEEP.map((k) => [k, data[k]])))); } catch (err) {}
  $("go").disabled = true;
  const r = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const j = await r.json();
  if (!r.ok) { status(j.error, true); $("go").disabled = false; return; }
  since = 0; log.textContent = ""; show(null);
  status("Starting " + j.run_id + "...");
  poll();
};

loadHistory().then(poll);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
