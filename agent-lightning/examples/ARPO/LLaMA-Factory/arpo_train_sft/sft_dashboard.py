#!/usr/bin/env python3
"""ARPO SFT 看板：loss 曲线 + 训练中 LiveGenCallback 写出的模型 generate。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

LOSS_RE = re.compile(
    r"\{'loss': (?P<loss>[-+eE0-9.]+),\s*"
    r"'grad_norm': (?P<grad>[-+eE0-9.naninfNANINF]+),\s*"
    r"'learning_rate': (?P<lr>[-+eE0-9.]+),\s*"
    r"'epoch': (?P<epoch>[-+eE0-9.]+)\}"
)
STEPS_RE = re.compile(r"Total optimization steps = ([\d,]+)")
DONE_RE = re.compile(r"train_runtime")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
LIVE_TAG_RE = re.compile(r"\[live_gen\]")


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid(path: Path) -> int:
    try:
        return int(path.read_text().strip().split()[0])
    except (OSError, ValueError):
        return 0


def _elapsed(pid: int) -> float | None:
    stat = Path(f"/proc/{pid}/stat")
    try:
        fields = stat.read_text().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return max(0.0, uptime - start_ticks / hz)
    except (OSError, IndexError, ValueError):
        return None


def _gpus() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_used_mib": int(parts[1]),
                "memory_total_mib": int(parts[2]),
                "util": int(parts[3]),
            }
        )
    return rows


def _parse_log(path: Path, logging_steps: int) -> dict:
    if not path.is_file():
        return {"points": [], "total_steps": None, "finished": False, "tail": "", "mtime": None}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"points": [], "total_steps": None, "finished": False, "tail": "", "mtime": None}
    points = []
    for match in LOSS_RE.finditer(text):
        grad = match.group("grad").lower()
        points.append(
            {
                "step": (len(points) + 1) * logging_steps,
                "loss": float(match.group("loss")),
                "grad_norm": None if grad == "nan" else float(match.group("grad")),
                "learning_rate": float(match.group("lr")),
                "epoch": float(match.group("epoch")),
            }
        )
    total = None
    found = STEPS_RE.findall(text)
    if found:
        total = int(found[-1].replace(",", ""))
    lines = []
    for raw in text.splitlines():
        piece = raw.split("\r")[-1].strip()
        if piece:
            lines.append(piece)
    return {
        "points": points,
        "total_steps": total,
        "finished": bool(DONE_RE.search(text)),
        "tail": "\n".join(lines[-40:]),
        "mtime": path.stat().st_mtime,
        "live_gen_log_lines": [ln for ln in lines if LIVE_TAG_RE.search(ln)][-8:],
    }


def _read_live(output_dir: Path) -> dict:
    latest = output_dir / "live_generation_latest.json"
    hist = output_dir / "live_generations.jsonl"
    empty = {
        "available": False,
        "step": None,
        "ok": None,
        "secs": None,
        "samples": [],
        "history_steps": [],
        "note": "等待训练 LiveGen：每 N step 会把当前权重的 generate 写到这里",
    }
    history: list[dict] = []
    if hist.is_file():
        try:
            for line in hist.read_text(encoding="utf-8").splitlines()[-12:]:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("step") is not None:
                    history.append(rec)
        except OSError:
            history = []
    if latest.is_file():
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
            data["available"] = True
            data["history"] = history
            data["history_steps"] = [h.get("step") for h in history if h.get("step") is not None]
            return data
        except (OSError, json.JSONDecodeError) as exc:
            empty["note"] = f"读取 live_generation_latest.json 失败: {exc}"
            return empty
    meta = output_dir / "live_gen_meta.json"
    if meta.is_file():
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            empty["note"] = (
                f"LiveGen 已启用（every={info.get('every_n_steps')}），"
                f"还没有写出第一轮 generate，请等到第 {info.get('every_n_steps')} step。"
            )
        except (OSError, json.JSONDecodeError):
            pass
    return empty


def status(log_path: Path, pid_path: Path, output_dir: Path, logging_steps: int, expected_steps: int) -> dict:
    pid = _read_pid(pid_path)
    running = _alive(pid)
    parsed = _parse_log(log_path, logging_steps)
    points = parsed["points"]
    total = parsed["total_steps"] or expected_steps
    step = points[-1]["step"] if points else 0
    elapsed = _elapsed(pid) if running else None
    eta = None
    if running and elapsed and step > 0 and total > step:
        eta = elapsed / step * (total - step)
    if parsed["finished"]:
        phase = "已结束"
    elif running and step == 0:
        phase = "正在加载"
    elif running:
        phase = "训练中"
    elif step > 0:
        phase = "进程已退出"
    else:
        phase = "未启动"
    age = None
    if parsed["mtime"]:
        age = max(0.0, time.time() - parsed["mtime"])
    live = _read_live(output_dir)
    return {
        "phase": phase,
        "pid": pid or None,
        "running": running,
        "elapsed_sec": elapsed,
        "eta_sec": eta,
        "step": step,
        "total_steps": total,
        "log_age_sec": age,
        "points": points,
        "gpus": _gpus(),
        "tail": parsed["tail"],
        "live_gen_log_lines": parsed.get("live_gen_log_lines") or [],
        "live": live,
    }


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ARPO SFT</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; background: #10141a; color: #e7ecf3; }
  header { padding: 16px 20px 8px; display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }
  h1 { font-size: 18px; margin: 0; font-weight: 600; }
  .muted { color: #93a0b4; }
  main { padding: 0 20px 24px; display: grid; gap: 12px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; }
  .card, .panel { background: #1a212b; border: 1px solid #2a3442; border-radius: 10px; padding: 12px 14px; }
  .card b { display: block; font-size: 20px; font-variant-numeric: tabular-nums; }
  .charts { display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 8px; }
  svg { width: 100%; height: 180px; background: #12181f; border-radius: 8px; }
  pre { margin: 0; max-height: 320px; overflow: auto; white-space: pre-wrap; word-break: break-word; color: #c5d0de; font-size: 12px; }
  .phase { padding: 2px 8px; border-radius: 999px; background: #243044; }
  .io { display: grid; grid-template-columns: 1fr 1.2fr 1fr; gap: 8px; }
  .io pre { max-height: 520px; background: #12181f; border-radius: 8px; padding: 10px; }
  .tag { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 999px; margin-left: 6px; }
  .tag-gt { background: #3a2a12; color: #f0c674; }
  .tag-live { background: #14301f; color: #7ddea0; }
  .tag-wait { background: #2a2030; color: #d7a0e8; }
  .tag-in { background: #1c2a3a; color: #8ec0ff; }
  .answer { margin: 8px 0 0; padding: 8px 10px; background: #243014; border: 1px solid #3f4f22; border-radius: 8px; color: #d8f0a8; }
  .warn { color: #f0c674; margin: 8px 0 0; }
  .row { display: flex; justify-content: space-between; gap: 8px; align-items: center; flex-wrap: wrap; }
  button { background: #243044; color: #e7ecf3; border: 1px solid #3a4a60; border-radius: 8px; padding: 4px 10px; cursor: pointer; }
  .sample-tabs { display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }
  .sample-tabs button.active { background: #3a5a40; border-color: #5a8a60; }
  .live-list { display: grid; gap: 10px; }
  .live-card { background: #12181f; border: 1px solid #2a3442; border-radius: 8px; padding: 10px 12px; }
  .live-card h3 { margin: 0 0 6px; font-size: 13px; font-weight: 600; color: #c5d0de; }
  .live-card .q { color: #8ec0ff; margin: 0 0 8px; white-space: pre-wrap; word-break: break-word; max-height: 72px; overflow: auto; }
  .live-card pre { max-height: 220px; }
  @media (max-width: 1100px) { .io { grid-template-columns: 1fr; } }
  @media (max-width: 900px) { .charts { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>ARPO 冷启动 SFT · 训练中 Live Generate</h1>
  <div><span id="phase" class="phase">连接中</span> <span id="updated" class="muted"></span></div>
</header>
<main>
  <section class="cards" id="cards"></section>
  <section class="charts">
    <div class="panel"><div class="muted">loss</div><svg id="loss"></svg></div>
    <div class="panel"><div class="muted">learning rate</div><svg id="lr"></svg></div>
    <div class="panel"><div class="muted">grad norm</div><svg id="grad"></svg></div>
  </section>
  <section class="panel">
    <div class="row">
      <div class="muted">训练过程模型输出（LiveGenCallback · 当前权重 generate）</div>
      <span id="sample-meta" class="muted"></span>
    </div>
    <p class="warn" id="sample-warn">等待第一轮 live generate…</p>
    <div class="sample-tabs" id="step-tabs"></div>
    <div class="live-list" id="live-list">等待中</div>
  </section>
  <section class="panel"><div class="muted">日志尾部</div><pre id="tail">尚无日志</pre></section>
</main>
<script>
const fmt = (sec) => {
  if (sec == null || !isFinite(sec)) return "—";
  sec = Math.round(sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h) return h + "小时" + m + "分";
  if (m) return m + "分" + s + "秒";
  return s + "秒";
};
function line(svg, points, key) {
  const w = 640, h = 180, pad = 16;
  const vals = points.map(p => p[key]).filter(v => v != null && isFinite(v));
  if (!vals.length) { svg.innerHTML = ""; return; }
  const xs = points.filter(p => p[key] != null).map(p => p.step);
  const ys = points.filter(p => p[key] != null).map(p => p[key]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const X = (x) => pad + (xmax === xmin ? (w - 2 * pad) / 2 : (x - xmin) / (xmax - xmin) * (w - 2 * pad));
  const Y = (y) => h - pad - (y - ymin) / (ymax - ymin) * (h - 2 * pad);
  const d = xs.map((x, i) => (i ? "L" : "M") + X(x).toFixed(1) + " " + Y(ys[i]).toFixed(1)).join(" ");
  svg.setAttribute("viewBox", "0 0 " + w + " " + h);
  svg.innerHTML = '<path d="' + d + '" fill="none" stroke="#7eb6ff" stroke-width="2"/>'
    + '<text x="' + pad + '" y="14" fill="#93a0b4" font-size="12">' + ymax.toPrecision(3) + "</text>"
    + '<text x="' + pad + '" y="' + (h - 4) + '" fill="#93a0b4" font-size="12">' + ymin.toPrecision(3) + "</text>";
}
let livePayload = null;
let viewStep = null;
function esc(s) {
  return String(s || "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}
function recordForStep(live, step) {
  const hist = live.history || [];
  for (let i = hist.length - 1; i >= 0; i--) {
    if (hist[i].step === step) return hist[i];
  }
  if (live.step === step) return live;
  return live;
}
function renderLive() {
  const live = livePayload;
  const list = document.getElementById("live-list");
  const tabs = document.getElementById("step-tabs");
  if (!live || !live.available) {
    document.getElementById("sample-warn").textContent = (live && live.note) || "等待 LiveGen…";
    document.getElementById("sample-meta").textContent = "";
    tabs.innerHTML = "";
    list.textContent = "等待中";
    return;
  }
  const rec = recordForStep(live, viewStep == null ? live.step : viewStep);
  const samples = rec.samples || [];
  document.getElementById("sample-warn").textContent =
    "本页一次列出该 step 的全部抽检 · 耗时 " + (rec.secs || "?") + "s · 共 " + samples.length + " 条（题目会随 step 轮换）";
  document.getElementById("sample-meta").textContent = samples.length + " 条 @ step " + rec.step;
  const steps = (live.history_steps && live.history_steps.length) ? live.history_steps : [live.step];
  const cur = rec.step;
  tabs.innerHTML = steps.map(st =>
    '<button type="button" data-step="' + st + '" class="' + (st === cur ? "active" : "") + '">step ' + st + "</button>"
  ).join("");
  tabs.querySelectorAll("button").forEach(btn => btn.onclick = () => {
    viewStep = parseInt(btn.getAttribute("data-step"), 10);
    renderLive();
  });
  if (!rec.ok) {
    list.innerHTML = '<div class="live-card">generate 失败: ' + esc(rec.error || "") + "</div>";
    return;
  }
  list.innerHTML = samples.map((s, i) => {
    const q = (s.user || "").slice(0, 280);
    const tag = s.stopped_at ? ("停在 " + s.stopped_at) : (s.truncated ? "截断" : (s.n_tokens + " tok"));
    const think = s.has_think ? "含 &lt;think&gt;" : "无 think";
    return '<div class="live-card"><h3>样本 ' + (i + 1) +
      ' <span class="tag tag-live">' + tag + "</span> " +
      '<span class="tag tag-in">' + think + "</span></h3>" +
      '<p class="q">' + esc(q) + "</p>" +
      '<div class="muted">模型输出</div><pre>' + esc(s.model_output || "（空）") + "</pre>" +
      '<div class="muted" style="margin-top:8px">GT 摘要</div><pre>' + esc(s.gold_head || "（空）") +
      "</pre></div>";
  }).join("") || "无样本";
}
async function tick() {
  const res = await fetch("/api/status", {cache: "no-store"});
  const data = await res.json();
  document.getElementById("phase").textContent = data.phase;
  document.getElementById("updated").textContent = "更新于 " + new Date().toLocaleTimeString();
  const last = data.points.length ? data.points[data.points.length - 1] : null;
  const gpus = (data.gpus || []).map(g => "GPU" + g.index + " " + g.memory_used_mib + "/" + g.memory_total_mib + " MiB · " + g.util + "%").join(" ｜ ") || "无 GPU 信息";
  const liveStep = data.live && data.live.available ? data.live.step : "—";
  const cards = [
    ["阶段", data.phase],
    ["步数", data.step + " / " + data.total_steps],
    ["loss", last ? last.loss.toFixed(4) : "尚无"],
    ["学习率", last ? last.learning_rate.toExponential(2) : "尚无"],
    ["LiveGen step", liveStep],
    ["已运行", fmt(data.elapsed_sec)],
    ["预计剩余", data.step ? fmt(data.eta_sec) : "等第一个 loss"],
    ["显存", gpus],
  ];
  document.getElementById("cards").innerHTML = cards.map(([k, v]) =>
    '<div class="card"><span class="muted">' + k + "</span><b>" + v + "</b></div>").join("");
  line(document.getElementById("loss"), data.points, "loss");
  line(document.getElementById("lr"), data.points, "learning_rate");
  line(document.getElementById("grad"), data.points, "grad_norm");
  const extra = (data.live_gen_log_lines || []).join("\\n");
  document.getElementById("tail").textContent =
    (data.tail || "") + (extra ? ("\\n---- live_gen ----\\n" + extra) : "");
  const prevStep = livePayload && livePayload.step;
  livePayload = data.live;
  if (viewStep == null || viewStep === prevStep) viewStep = data.live && data.live.step;
  renderLive();
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--expected-steps", type=int, default=10233)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    log_path = Path(args.log)
    pid_path = Path(args.pid_file)
    output_dir = Path(args.output_dir) if args.output_dir else log_path.parent

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *log_args) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                payload = status(log_path, pid_path, output_dir, args.logging_steps, args.expected_steps)
                self._send(200, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")
                return
            if parsed.path == "/api/live":
                payload = _read_live(output_dir)
                self._send(200, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")
                return
            if parsed.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"sft dashboard http://127.0.0.1:{args.port} log={log_path} output_dir={output_dir}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
