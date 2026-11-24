"""Arrangement audition interface for the score-following accompaniment.

A small dependency-free web UI: the score-following kernel (estimator +
scheduler) runs server-side and drives every control - the score cursor,
the accompaniment playhead, stall indicators and the sound the page plays
all come from kernel responses, never from browser-side guesses. Kernel
failures are reported back into the page, which halts the transport until
the session is reset.

Launch:
    python -m score_follow.webui --port 8010
then open http://127.0.0.1:8010/ in a browser. Pick a score and a kernel
("optimized" is the selected budgeted update, "baseline" the pre-speedup
order), press "播放演奏" to actually play a saved performance into the
kernel, or play live with the on-screen keys (QWERTY row a w s e d f ...).
Toggle "跟谱" to compare the following accompaniment against a fixed-tempo
one: with follow off the accompaniment marches at the score tempo whatever
the performer does, and the two playheads drift apart on the staff.

API (all JSON):
    GET  /                     the page
    GET  /api/trajectories     saved performances under --data
    GET  /api/trajectory?name= one trajectory file (name must be listed)
    POST /api/session          {score, kernel, follow} -> fresh session
    POST /api/event            {kind: on|off, pitch: 0..127}
    POST /api/tick             clock update; returns due actions
    POST /api/follow           {enabled: bool}
    POST /api/reset            back to a fresh kernel with the same config
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .estimator import Estimator
from .scheduler import Scheduler, SystemClock
from .score import Score, demo_score
from .benchmark import bench_score
from .types import Estimate, NoteEvent

DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)

SCORES = {
    "demo": demo_score,
    "bench": bench_score,
}

# kernel variants: the selected budgeted update and the pre-speedup order,
# so the page can A/B the accompaniment change on the same input
KERNELS = {
    "optimized": {},
    "baseline": {
        "local_budget": None,
        "far_jump_budget": None,
        "far_jump_proposers": None,
    },
}


class FollowKernel:
    """The running accompaniment kernel behind the audition page."""

    def __init__(self, score_name: str = "demo", kernel: str = "optimized",
                 follow: bool = True) -> None:
        self.score_name = score_name
        self.kernel_name = kernel
        self.follow = follow
        self._build()

    def _build(self) -> None:
        self.score: Score = SCORES[self.score_name]()
        self.clock = SystemClock()
        self.estimator = Estimator(self.score, **KERNELS[self.kernel_name])
        self.scheduler = Scheduler(self.score, self.clock)
        self.tempo0 = self.score.initial_tempo_bpm / 60.0
        self._t0 = self.clock.now()

    def reset(self) -> None:
        self._build()

    def set_follow(self, enabled: bool) -> None:
        self.follow = bool(enabled)

    # ------------------------------------------------------------------ #
    def _fixed_estimate(self, now: float) -> Estimate:
        """Fixed-tempo marching estimate used when follow is switched off:
        the accompaniment advances at the score tempo whatever arrived."""
        return Estimate(
            time=now,
            positions=(),
            best_beat=max(0.0, (now - self._t0)) * self.tempo0,
            best_index=0,
            tempo=self.tempo0,
            confidence=1.0,
            margin=1.0,
            expected_index=0,
            event_count=max(1, self.estimator.event_count),
            is_onset=False,
        )

    def _scheduling_estimate(self, now: float) -> Estimate:
        if self.follow:
            return self.estimator.tick(now)
        return self._fixed_estimate(now)

    def handle_event(self, kind: str, pitch: int) -> dict:
        estimate = self.estimator.handle_event(NoteEvent(self.clock.now(), kind, pitch))
        actions = self.scheduler.update(self._scheduling_estimate(self.clock.now()))
        return self._payload(estimate, actions)

    def tick(self) -> dict:
        now = self.clock.now()
        tracker = self.estimator.tick(now)
        actions = self.scheduler.poll(self._scheduling_estimate(now))
        return self._payload(tracker, actions)

    # ------------------------------------------------------------------ #
    def _payload(self, tracker: Estimate, actions) -> dict:
        return {
            "now": round(self.clock.now(), 4),
            "follow": self.follow,
            "kernel": self.kernel_name,
            "score": self.score.name,
            "cursor": _estimate_json(tracker),
            "playhead_beat": round(
                tracker.best_beat if self.follow
                else max(0.0, self.clock.now() - self._t0) * self.tempo0, 4
            ),
            "actions": [_action_json(a, e) for a, e in actions],
        }


def _estimate_json(est: Estimate) -> dict:
    return {
        "time": round(est.time, 4),
        "best_beat": round(est.best_beat, 4),
        "tempo": round(est.tempo, 4),
        "confidence": round(est.confidence, 4),
        "margin": round(est.margin, 4),
        "expected_index": est.expected_index,
        "event_count": est.event_count,
        "positions": [[round(b, 3), round(w, 4)] for b, w in est.positions[:8]],
    }


def _action_json(action: str, entry) -> dict:
    return {
        "action": action,
        "pitch": entry.note.pitch,
        "velocity": entry.note.velocity,
        "velocity_scale": round(entry.velocity_scale, 3),
        "target_beat": entry.note.start,
        "play_time": round(entry.play_time, 4),
        "off_time": round(entry.off_time, 4),
    }


def _score_json(score: Score) -> dict:
    return {
        "name": score.name,
        "initial_tempo_bpm": score.initial_tempo_bpm,
        "length_beats": score.length_beats,
        "solo": [[n.start, n.duration, n.pitch] for n in score.solo],
        "accompaniment": [[n.start, n.duration, n.pitch] for n in score.accompaniment],
        "repeats": [list(r) for r in score.repeats],
    }


_PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>跟谱伴奏 · 编曲试听</title>
<style>
 body { font: 14px/1.5 system-ui, sans-serif; margin: 18px; background:#14171c; color:#dde3ea; }
 h1 { font-size: 18px; margin: 0 0 10px; }
 .row { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin:8px 0; }
 button, select { background:#232a33; color:#dde3ea; border:1px solid #3a4450; border-radius:4px; padding:4px 10px; }
 button:disabled { opacity:.4; }
 canvas { background:#0d1014; border:1px solid #2a323d; border-radius:6px; }
 #err { background:#4a1d1d; border:1px solid #7a2e2e; padding:6px 10px; border-radius:4px; display:none; }
 #log { font:11px/1.4 monospace; color:#8fa1b3; height:120px; overflow-y:auto; background:#0d1014;
        border:1px solid #2a323d; border-radius:6px; padding:6px; white-space:pre-wrap; }
 .meter { width:160px; height:10px; background:#232a33; border-radius:5px; overflow:hidden; }
 .meter > div { height:100%; background:#4da3ff; width:0%; }
 .key { width:34px; height:56px; margin:1px; font-size:11px; }
 label { user-select:none; }
</style>
</head>
<body>
<h1>跟谱伴奏 · 编曲试听界面</h1>
<div id="err"></div>
<div class="row">
  <label>乐谱 <select id="score"><option value="demo">demo_repeat_phrase</option><option value="bench">bench_sections</option></select></label>
  <label>内核 <select id="kernel"><option value="optimized">优化（远跳预算档）</option><option value="baseline">参照（变更前）</option></select></label>
  <label><input type="checkbox" id="follow" checked> 跟谱</label>
  <button id="newsession">建立会话</button>
  <button id="reset">重置</button>
  <span id="status">未连接</span>
</div>
<div class="row">
  <label>演奏输入 <select id="traj"></select></label>
  <button id="play">播放演奏</button>
  <button id="stop" disabled>停止</button>
  <span>置信度</span><div class="meter"><div id="conf"></div></div>
  <span id="beat">beat 0.0</span>
</div>
<canvas id="staff" width="960" height="320"></canvas>
<div class="row" id="keys"></div>
<div id="log"></div>
<script>
const $ = id => document.getElementById(id);
let session = null, scoreData = null, timers = [], audioCtx = null;

function showErr(msg) { const e = $("err"); e.style.display = "block"; e.textContent = "内核反馈: " + msg; }
function clearErr() { $("err").style.display = "none"; }
function logLine(s) { const l = $("log"); l.textContent += s + "\\n"; l.scrollTop = l.scrollHeight; }

async function api(path, body) {
  const opt = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { showErr(data.error || (r.status + " " + path)); throw new Error(data.error || r.status); }
  clearErr();
  return data;
}

function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
}
function tone(freq, dur, vel, when) {
  if (!audioCtx) return;
  const t = Math.max(audioCtx.currentTime, when || audioCtx.currentTime);
  const o = audioCtx.createOscillator(), g = audioCtx.createGain();
  o.type = "triangle"; o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(Math.max(0.001, vel), t + 0.015);
  g.gain.exponentialRampToValueAtTime(0.0001, t + Math.max(0.08, dur));
  o.connect(g).connect(audioCtx.destination);
  o.start(t); o.stop(t + Math.max(0.1, dur) + 0.05);
}
const freqOf = p => 440 * Math.pow(2, (p - 69) / 12);

async function newSession() {
  stopPlayback();
  const res = await api("/api/session", {score: $("score").value, kernel: $("kernel").value, follow: $("follow").checked});
  session = res.session; scoreData = res.score_data;
  $("status").textContent = "会话中 · " + res.score + " · 内核 " + res.kernel;
  drawStaff();
  logLine("会话建立: score=" + res.score + " kernel=" + res.kernel + " follow=" + res.follow);
}

function drawStaff(cursorBeat, playheadBeat, positions) {
  const c = $("staff"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (!scoreData) return;
  const pitches = scoreData.solo.map(n => n[2]);
  const lo = Math.min(...pitches) - 3, hi = Math.max(...pitches) + 3;
  const x = b => 30 + b / Math.max(1, scoreData.length_beats) * (c.width - 50);
  const y = p => c.height - 20 - (p - lo) / (hi - lo) * (c.height - 40);
  ctx.strokeStyle = "#222a34";
  for (let b = 0; b <= scoreData.length_beats; b += 4) {
    ctx.beginPath(); ctx.moveTo(x(b), 8); ctx.lineTo(x(b), c.height - 8); ctx.stroke();
  }
  for (const n of scoreData.accompaniment) {
    ctx.fillStyle = "rgba(120,140,160,.25)";
    ctx.fillRect(x(n[0]), y(n[2]) - 3, Math.max(2, x(n[0] + n[1]) - x(n[0])), 6);
  }
  for (const n of scoreData.solo) {
    ctx.fillStyle = "#3f6f9f";
    ctx.fillRect(x(n[0]), y(n[2]) - 4, Math.max(2, x(n[0] + n[1]) - x(n[0])), 8);
  }
  if (positions) for (const [b, w] of positions) {
    ctx.strokeStyle = "rgba(255,200,80," + Math.min(0.9, w) + ")";
    ctx.beginPath(); ctx.moveTo(x(b), c.height - 14); ctx.lineTo(x(b), c.height - 4); ctx.stroke();
  }
  if (playheadBeat !== undefined && playheadBeat !== cursorBeat) {
    ctx.strokeStyle = "#7ac74f"; ctx.beginPath();
    ctx.moveTo(x(playheadBeat), 4); ctx.lineTo(x(playheadBeat), c.height - 4); ctx.stroke();
  }
  if (cursorBeat !== undefined) {
    ctx.strokeStyle = "#ff5f5f"; ctx.beginPath();
    ctx.moveTo(x(cursorBeat), 4); ctx.lineTo(x(cursorBeat), c.height - 4); ctx.stroke();
  }
}

function render(d) {
  drawStaff(d.cursor.best_beat, d.playhead_beat, d.cursor.positions);
  $("conf").style.width = Math.round(d.cursor.confidence * 100) + "%";
  $("beat").textContent = "beat " + d.cursor.best_beat.toFixed(2) + (d.follow ? "" : " · 固定速度播放头 " + d.playhead_beat.toFixed(2));
  for (const a of d.actions) {
    if (a.action === "play") {
      tone(freqOf(a.pitch), Math.max(0.1, a.off_time - a.play_time), 0.22 * a.velocity_scale);
      logLine("♪ 伴奏 " + a.pitch + " @beat " + a.target_beat + " vel×" + a.velocity_scale);
    } else {
      logLine("… 等待（置信度不足） beat " + a.target_beat);
    }
  }
}

async function sendEvent(kind, pitch) {
  ensureAudio();
  if (kind === "on") tone(freqOf(pitch), 0.35, 0.3);
  try { render(await api("/api/event", {kind, pitch})); }
  catch (e) { stopPlayback(); }
}

let pollTimer = null;
async function poll() {
  if (!session) return;
  try { render(await api("/api/tick", {})); } catch (e) { /* banner shown */ }
}

async function playTrajectory() {
  if (!session) await newSession();
  ensureAudio();
  const name = $("traj").value;
  if (!name) return;
  const data = await api("/api/trajectory?name=" + encodeURIComponent(name));
  stopPlayback();
  $("play").disabled = true; $("stop").disabled = false;
  logLine("播放演奏 " + name + "（" + data.events.length + " 事件）");
  for (const ev of data.events) {
    timers.push(setTimeout(() => sendEvent(ev.kind, ev.pitch), ev.time * 1000));
  }
  timers.push(setTimeout(() => { $("play").disabled = false; $("stop").disabled = true; logLine("演奏结束"); },
    data.events.length ? data.events[data.events.length - 1].time * 1000 + 400 : 400));
}
function stopPlayback() {
  for (const t of timers) clearTimeout(t);
  timers = [];
  $("play").disabled = false; $("stop").disabled = true;
}

const KEYMAP = {a:60,w:61,s:62,e:63,d:64,f:65,t:66,g:67,y:68,h:69,u:70,j:71,k:72,o:73,l:74,p:75,";":76};
const held = new Set();
function buildKeys() {
  const div = $("keys");
  for (const [k, p] of Object.entries(KEYMAP)) {
    const b = document.createElement("button");
    b.className = "key"; b.textContent = k + " · " + p;
    b.onmousedown = () => keyEvent(k);
    div.appendChild(b);
  }
}
function keyEvent(k) {
  const p = KEYMAP[k];
  if (p === undefined) return;
  if (held.has(p)) { held.delete(p); sendEvent("off", p); }
  else { held.add(p); sendEvent("on", p); }
}
document.addEventListener("keydown", e => { if (!e.repeat && KEYMAP[e.key] !== undefined) keyEvent(e.key); });

$("newsession").onclick = () => newSession().catch(() => {});
$("reset").onclick = () => api("/api/reset", {}).then(d => { logLine("内核已重置"); }).catch(() => {});
$("follow").onchange = () => api("/api/follow", {enabled: $("follow").checked})
  .then(d => logLine("跟谱 " + (d.follow ? "开" : "关（固定速度伴奏）"))).catch(() => {});
$("play").onclick = () => playTrajectory().catch(() => {});
$("stop").onclick = () => stopPlayback();

async function boot() {
  buildKeys();
  const list = await api("/api/trajectories");
  for (const n of list.items) {
    const o = document.createElement("option");
    o.value = n; o.textContent = n;
    $("traj").appendChild(o);
  }
  await newSession();
  pollTimer = setInterval(poll, 50);
}
boot().catch(e => showErr(e.message || e));
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "ScoreFollowUI/1.0"

    # -- helpers -------------------------------------------------------- #
    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 1_000_000:
            raise ValueError("bad request body")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    # -- routing -------------------------------------------------------- #
    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            body = _PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/trajectories":
            self._send_json({"items": self.server.list_trajectories()})
            return
        if self.path.startswith("/api/trajectory?name="):
            name = self.path.split("name=", 1)[1]
            from urllib.parse import unquote
            name = unquote(name)
            data = self.server.load_trajectory(name)
            if data is None:
                self._send_json({"error": f"unknown trajectory {name!r}"}, 404)
                return
            self._send_json(data)
            return
        self._send_json({"error": f"unknown route {self.path}"}, 404)

    def do_POST(self) -> None:
        try:
            body = self._read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": f"bad request: {exc}"}, 400)
            return
        try:
            if self.path == "/api/session":
                self._api_session(body)
            elif self.path == "/api/event":
                self._api_event(body)
            elif self.path == "/api/tick":
                self._with_kernel(lambda k: self._send_json(k.tick()))
            elif self.path == "/api/follow":
                self._api_follow(body)
            elif self.path == "/api/reset":
                self._with_kernel(lambda k: (k.reset(), self._send_json({"ok": True})))
            else:
                self._send_json({"error": f"unknown route {self.path}"}, 404)
        except Exception as exc:  # kernel failure feeds back to the page
            self._send_json({"error": f"kernel failure: {exc}"}, 500)

    # -- endpoints ------------------------------------------------------ #
    def _with_kernel(self, fn):
        with self.server.lock:
            fn(self.server.kernel)

    def _api_session(self, body: dict) -> None:
        score_name = body.get("score", "demo")
        kernel_name = body.get("kernel", "optimized")
        follow = bool(body.get("follow", True))
        if score_name not in SCORES:
            self._send_json({"error": f"unknown score {score_name!r}"}, 400)
            return
        if kernel_name not in KERNELS:
            self._send_json({"error": f"unknown kernel {kernel_name!r}"}, 400)
            return
        with self.server.lock:
            self.server.kernel = FollowKernel(score_name, kernel_name, follow)
            kernel = self.server.kernel
            self._send_json({
                "session": id(kernel),
                "score": kernel.score.name,
                "kernel": kernel_name,
                "follow": follow,
                "score_data": _score_json(kernel.score),
            })

    def _api_event(self, body: dict) -> None:
        kind = body.get("kind")
        pitch = body.get("pitch")
        if kind not in ("on", "off"):
            self._send_json({"error": f"bad event kind {kind!r}"}, 400)
            return
        if not isinstance(pitch, int) or not 0 <= pitch <= 127:
            self._send_json({"error": f"bad pitch {pitch!r}"}, 400)
            return
        self._with_kernel(lambda k: self._send_json(k.handle_event(kind, pitch)))

    def _api_follow(self, body: dict) -> None:
        enabled = bool(body.get("enabled", True))
        def go(kernel):
            kernel.set_follow(enabled)
            self._send_json({"follow": kernel.follow})
        self._with_kernel(go)


class AuditionServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, data_dir: str = DATA_DIR,
                 score_name: str = "demo", kernel: str = "optimized") -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.data_dir = data_dir
        self.lock = threading.Lock()
        self.kernel = FollowKernel(score_name, kernel, True)

    def list_trajectories(self) -> list[str]:
        out = []
        for root, _dirs, files in os.walk(self.data_dir):
            for f in sorted(files):
                if f.endswith("_trajectory.json"):
                    out.append(os.path.relpath(os.path.join(root, f), self.data_dir))
        return sorted(out)

    def load_trajectory(self, name: str) -> dict | None:
        # only names that came from list_trajectories are loadable
        if name not in set(self.list_trajectories()):
            return None
        with open(os.path.join(self.data_dir, name), encoding="utf-8") as fh:
            return json.load(fh)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="score-follow-webui",
                                description="arrangement audition interface")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--data", default=DATA_DIR, help="directory with *_trajectory.json")
    p.add_argument("--score", default="demo", choices=sorted(SCORES))
    p.add_argument("--kernel", default="optimized", choices=sorted(KERNELS))
    args = p.parse_args(argv)
    server = AuditionServer(args.port, args.data, args.score, args.kernel)
    print(f"编曲试听界面已启动: http://127.0.0.1:{server.server_port}/")
    print("在浏览器中打开；选择乐谱与内核后按“播放演奏”或用键盘 a w s e d f ... 演奏。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
