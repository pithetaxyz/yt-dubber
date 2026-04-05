"""
YT-Dubber Web Interface
Run with: python app.py
Then open: http://localhost:5000
"""

import sys
import io
import json
import queue
import threading
from pathlib import Path

from flask import Flask, Response, request, jsonify

app = Flask(__name__)

# ── Job state ────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_state = {
    "status": "idle",   # idle | running | done | error
    "log": [],
    "result": None,
    "error": None,
}
_log_q: queue.Queue = queue.Queue()


# ── Stdout/stderr capture ────────────────────────────────────────────────────
class _Capture(io.TextIOBase):
    """Tees all writes to both the original stream and the SSE log queue."""

    def __init__(self, original):
        self._orig = original
        self._buf = ""

    def write(self, s: str) -> int:
        self._orig.write(s)
        self._orig.flush()
        # Buffer and flush on newlines; discard bare carriage-returns
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r").strip()
            if line:
                _log_q.put(line)
                with _lock:
                    _state["log"].append(line)
        return len(s)

    def flush(self):
        self._orig.flush()

    def reconfigure(self, **kwargs):
        pass  # no-op — dubber.py calls this at import time


# ── Pipeline thread ──────────────────────────────────────────────────────────
def _run(url, out_dir, whisper_size, voice, translator, do_upload):
    with _lock:
        _state.update(status="running", log=[], result=None, error=None)

    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Capture(orig_out)
    sys.stderr = _Capture(orig_err)

    try:
        import dubber
        out_path, title_en, source_url, transcript_path = dubber.run(
            url=url,
            out_dir=Path(out_dir),
            whisper_size=whisper_size,
            voice=voice or None,
            translator=translator,
        )

        result = {"output": str(out_path), "title": title_en}

        if do_upload:
            from uploader import upload_to_youtube
            video_id = upload_to_youtube(
                str(out_path),
                f"{title_en} [English Dubbed]",
                source_url=source_url,
                source_title=title_en,
                transcript_path=str(transcript_path),
            )
            result["youtube_id"] = video_id
            result["youtube_url"] = f"https://www.youtube.com/watch?v={video_id}"

        with _lock:
            _state.update(status="done", result=result)
        _log_q.put("__DONE__")

    except Exception as exc:
        import traceback
        sys.stdout.write(traceback.format_exc())
        with _lock:
            _state.update(status="error", error=str(exc))
        _log_q.put("__ERROR__")
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return _HTML


@app.route("/run", methods=["POST"])
def run_job():
    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "A job is already running."}), 409

    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL is required."}), 400

    # Drain stale queue entries
    while not _log_q.empty():
        try:
            _log_q.get_nowait()
        except queue.Empty:
            break

    threading.Thread(
        target=_run,
        args=(
            url,
            data.get("out_dir") or "output",
            data.get("whisper_size") or "large",
            data.get("voice") or "",
            data.get("translator") or "helsinki",
            bool(data.get("upload", True)),
        ),
        daemon=True,
    ).start()

    return jsonify({"status": "started"})


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint for live log lines."""
    def gen():
        while True:
            try:
                msg = _log_q.get(timeout=25)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg in ("__DONE__", "__ERROR__"):
                    break
            except queue.Empty:
                yield "data: \"__PING__\"\n\n"

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/status")
def status():
    with _lock:
        return jsonify(_state)


# ── Inline HTML UI ───────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YT-Dubber</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem; }
  h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; color: #fff; }
  .subtitle { color: #888; font-size: 0.85rem; margin-bottom: 2rem; }
  .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 1.5rem; width: 100%; max-width: 680px; margin-bottom: 1.25rem; }
  label { display: block; font-size: 0.8rem; color: #999; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em; }
  input[type=text], input[type=url], select {
    width: 100%; background: #111; border: 1px solid #333; border-radius: 8px;
    color: #e0e0e0; padding: 0.6rem 0.8rem; font-size: 0.95rem; outline: none;
    transition: border-color 0.15s;
  }
  input:focus, select:focus { border-color: #ff4444; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  .url-row { margin-bottom: 0; }
  .toggle-row { display: flex; align-items: center; gap: 0.6rem; margin-top: 1rem; }
  .toggle-row input[type=checkbox] { width: 16px; height: 16px; accent-color: #ff4444; cursor: pointer; }
  .toggle-row span { font-size: 0.9rem; color: #ccc; }
  button#runBtn {
    width: 100%; max-width: 680px; padding: 0.8rem;
    background: #e00; border: none; border-radius: 10px;
    color: #fff; font-size: 1rem; font-weight: 600; cursor: pointer;
    transition: background 0.15s, opacity 0.15s;
  }
  button#runBtn:hover:not(:disabled) { background: #c00; }
  button#runBtn:disabled { opacity: 0.45; cursor: default; }
  .console-card { width: 100%; max-width: 680px; }
  #console {
    background: #0a0a0a; border: 1px solid #2a2a2a; border-radius: 10px;
    padding: 1rem; height: 340px; overflow-y: auto;
    font-family: 'Consolas', 'Menlo', monospace; font-size: 0.78rem;
    line-height: 1.5; color: #b0ffb0;
    white-space: pre-wrap; word-break: break-all;
  }
  #console .err { color: #ff8080; }
  #console .info { color: #80c8ff; }
  #console .step { color: #ffd080; font-weight: bold; }
  .result-card { width: 100%; max-width: 680px; background: #0f1f0f; border: 1px solid #1a3a1a; border-radius: 12px; padding: 1.25rem; display: none; }
  .result-card h3 { color: #6f6; margin-bottom: 0.75rem; }
  .result-card a { color: #ff6666; text-decoration: none; }
  .result-card a:hover { text-decoration: underline; }
  .badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 99px; font-size: 0.72rem; font-weight: 600; margin-left: 0.5rem; }
  .badge.running { background: #333; color: #ffd080; }
  .badge.done { background: #1a3a1a; color: #6f6; }
  .badge.error { background: #3a1a1a; color: #f66; }
</style>
</head>
<body>

<h1>YT-Dubber <span id="badge" class="badge" style="display:none"></span></h1>
<p class="subtitle">Chinese YouTube &rarr; English dubbed video, auto-uploaded to your channel</p>

<div class="card">
  <div class="url-row">
    <label for="url">YouTube URL</label>
    <input type="url" id="url" placeholder="https://www.youtube.com/watch?v=..." autocomplete="off" spellcheck="false">
  </div>
  <div class="row3">
    <div>
      <label for="whisper">Whisper model</label>
      <select id="whisper">
        <option value="large" selected>large (best)</option>
        <option value="medium">medium</option>
        <option value="small">small</option>
        <option value="base">base</option>
        <option value="tiny">tiny (fast)</option>
      </select>
    </div>
    <div>
      <label for="translator">Translator</label>
      <select id="translator">
        <option value="helsinki" selected>Helsinki (fast)</option>
        <option value="gemma">Gemma-4B (better)</option>
      </select>
    </div>
    <div>
      <label for="voice">TTS voice (optional)</label>
      <input type="text" id="voice" placeholder="auto-detect gender">
    </div>
  </div>
  <div class="row">
    <div>
      <label for="outdir">Output folder</label>
      <input type="text" id="outdir" value="output">
    </div>
    <div class="toggle-row" style="padding-top:1.4rem">
      <input type="checkbox" id="upload" checked>
      <span>Upload to YouTube after dubbing</span>
    </div>
  </div>
</div>

<button id="runBtn" onclick="startJob()">Run Pipeline</button>

<div style="width:100%;max-width:680px;display:flex;justify-content:space-between;align-items:center;margin:1rem 0 0.4rem">
  <span style="font-size:0.8rem;color:#555;text-transform:uppercase;letter-spacing:0.05em">Live log</span>
  <button onclick="clearLog()" style="background:none;border:none;color:#555;font-size:0.75rem;cursor:pointer">clear</button>
</div>
<div class="console-card"><div id="console"></div></div>

<div class="result-card" id="resultCard">
  <h3>Done!</h3>
  <div id="resultBody"></div>
</div>

<script>
let evtSource = null;

function log(text, cls) {
  const c = document.getElementById('console');
  const div = document.createElement('div');
  if (cls) div.className = cls;
  // Detect step banners
  if (text.startsWith('Step ') || text.startsWith('  Step ')) div.className = 'step';
  div.textContent = text;
  c.appendChild(div);
  c.scrollTop = c.scrollHeight;
}

function clearLog() {
  document.getElementById('console').innerHTML = '';
}

function setBadge(status) {
  const b = document.getElementById('badge');
  if (!status) { b.style.display = 'none'; return; }
  b.style.display = '';
  b.className = 'badge ' + status;
  b.textContent = status;
}

async function startJob() {
  const url = document.getElementById('url').value.trim();
  if (!url) { alert('Please enter a YouTube URL.'); return; }

  document.getElementById('runBtn').disabled = true;
  document.getElementById('resultCard').style.display = 'none';
  setBadge('running');
  clearLog();
  log('Starting pipeline...', 'info');

  const payload = {
    url,
    whisper_size: document.getElementById('whisper').value,
    translator:   document.getElementById('translator').value,
    voice:        document.getElementById('voice').value.trim(),
    out_dir:      document.getElementById('outdir').value.trim() || 'output',
    upload:       document.getElementById('upload').checked,
  };

  const res = await fetch('/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    log('[ERROR] ' + data.error, 'err');
    document.getElementById('runBtn').disabled = false;
    setBadge('error');
    return;
  }

  // Open SSE stream
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/stream');
  evtSource.onmessage = async (e) => {
    const msg = JSON.parse(e.data);
    if (msg === '__PING__') return;
    if (msg === '__DONE__') {
      evtSource.close();
      setBadge('done');
      document.getElementById('runBtn').disabled = false;
      const st = await fetch('/status').then(r => r.json());
      showResult(st.result);
      return;
    }
    if (msg === '__ERROR__') {
      evtSource.close();
      setBadge('error');
      document.getElementById('runBtn').disabled = false;
      log('[Pipeline failed — see log above]', 'err');
      return;
    }
    log(msg);
  };
  evtSource.onerror = () => {
    log('[Connection lost]', 'err');
    document.getElementById('runBtn').disabled = false;
  };
}

function showResult(result) {
  if (!result) return;
  const card = document.getElementById('resultCard');
  const body = document.getElementById('resultBody');
  let html = `<p style="margin-bottom:0.5rem;color:#ccc">Output: <code style="color:#adf">${result.output}</code></p>`;
  if (result.title) html += `<p style="margin-bottom:0.5rem;color:#ccc">Title: <em>${result.title}</em></p>`;
  if (result.youtube_url) {
    html += `<p>YouTube: <a href="${result.youtube_url}" target="_blank">${result.youtube_url}</a></p>`;
  }
  body.innerHTML = html;
  card.style.display = 'block';
}

// On load, check if a job is already running and reconnect stream
window.onload = async () => {
  const st = await fetch('/status').then(r => r.json());
  if (st.status === 'running') {
    setBadge('running');
    document.getElementById('runBtn').disabled = true;
    for (const line of st.log) log(line);
    evtSource = new EventSource('/stream');
    evtSource.onmessage = async (e) => {
      const msg = JSON.parse(e.data);
      if (msg === '__PING__') return;
      if (msg === '__DONE__') {
        evtSource.close();
        setBadge('done');
        document.getElementById('runBtn').disabled = false;
        const s2 = await fetch('/status').then(r => r.json());
        showResult(s2.result);
        return;
      }
      if (msg === '__ERROR__') {
        evtSource.close();
        setBadge('error');
        document.getElementById('runBtn').disabled = false;
        return;
      }
      log(msg);
    };
  } else if (st.status === 'done') {
    setBadge('done');
    showResult(st.result);
  } else if (st.status === 'error') {
    setBadge('error');
    for (const line of st.log) log(line);
  }
};
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("YT-Dubber Web UI")
    print("Open: http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
