#!/usr/bin/env python3
"""Realtime monitor for scripts/agentflow_4algo_run.sh.

Modes:
  python scripts/agentflow_4algo_monitor.py              # HTTP dashboard (default :8765)
  python scripts/agentflow_4algo_monitor.py --cli         # terminal TUI (refresh every 2s)
  python scripts/agentflow_4algo_monitor.py --port 8765

Dashboard: http://127.0.0.1:8765/
API:       http://127.0.0.1:8765/api/status
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "agentflow_4algo_10step"
CKPT_ROOTS = [
    ROOT / "checkpoints" / "AgentLightning",
    ROOT / "mas" / "checkpoints" / "AgentLightning",
]
ALGOS = ["grpo", "arpo", "appo", "flow_grpo"]
METRIC_KEYS = [
    "training/reward",
    "val/reward",
    "actor/pg_loss",
    "actor/ppo_kl",
    "actor/entropy_loss",
    "training/n_rollouts",
    "training/n_triplets",
    "training/branch_local_count",
    "training/n_with_anchor",
]
REWARD_LINE_RE = re.compile(
    r"src=(?P<src>\S+)\s+pred=.*?gt='(?P<gt>[^']*)'.*?R=(?P<R>[-+eE0-9.]+).*?t=(?P<t>[-+eE0-9.]+)s",
    re.DOTALL,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" not in row:
            row["step"] = row.get("training/global_step")
        out.append(row)
    return out


def _metrics_path(algo: str) -> Path | None:
    names = [
        f"agentflow_4algo_{algo}_10step_{algo}/metrics.jsonl",
        f"agentflow_4algo_{algo}_10step/metrics.jsonl",
    ]
    for root in CKPT_ROOTS:
        for rel in names:
            p = root / rel
            if p.is_file():
                return p
    return None


def _ps_snapshot() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,etime,cmd", "--no-headers"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"script_running": False, "current_algo": None, "train_pids": [], "script_etime": None}

    script_running = False
    script_etime = None
    current_algo = None
    train_pids: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, etime, cmd = parts[0], parts[1], parts[2]
        if "agentflow_4algo_run.sh" in cmd and "grep" not in cmd:
            script_running = True
            script_etime = etime
        if "mas.train_tir_agent" in cmd or "train_tir_agent" in cmd:
            algo = None
            for a in ALGOS:
                if f"rl_{a}.yaml" in cmd or f"_{a}." in cmd:
                    algo = a
                    break
            train_pids.append({"pid": int(pid), "etime": etime, "algo": algo, "cmd": cmd[:160]})
            if algo and current_algo is None:
                current_algo = algo
    return {
        "script_running": script_running,
        "script_etime": script_etime,
        "current_algo": current_algo,
        "train_pids": train_pids,
    }


def _gpu_snapshot() -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 6:
            continue
        gpus.append(
            {
                "index": int(cols[0]),
                "util": float(cols[1]),
                "mem_used_mb": float(cols[2]),
                "mem_total_mb": float(cols[3]),
                "temp_c": float(cols[4]),
                "power_w": float(cols[5]),
            }
        )
    return gpus


def _tail_text(path: Path, n: int = 80) -> str:
    if not path.is_file():
        return ""
    try:
        # Efficient-ish tail for multi-MB logs
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 8192
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines()
            return "\n".join(lines[-n:])
    except OSError:
        return ""


def _recent_rewards(log_text: str, limit: int = 12) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for m in REWARD_LINE_RE.finditer(log_text):
        try:
            hits.append(
                {
                    "src": m.group("src"),
                    "gt": m.group("gt")[:40],
                    "R": float(m.group("R")),
                    "t": float(m.group("t")),
                }
            )
        except ValueError:
            continue
    return hits[-limit:]


def _algo_status(algo: str, current: str | None) -> dict[str, Any]:
    log_path = LOG_DIR / f"{algo}.log"
    mpath = _metrics_path(algo)
    rows = _read_jsonl(mpath) if mpath else []
    train_steps = [r for r in rows if r.get("training/reward") is not None or r.get("actor/pg_loss") is not None]
    last = rows[-1] if rows else None
    last_train = train_steps[-1] if train_steps else None
    log_mtime = log_path.stat().st_mtime if log_path.is_file() else None
    log_size = log_path.stat().st_size if log_path.is_file() else 0

    if current == algo:
        state = "running"
    elif rows:
        # finished if metrics exist and not current
        state = "done" if current is not None and ALGOS.index(algo) < ALGOS.index(current) else ("done" if current is None and log_size > 0 else "pending")
        if current is not None and ALGOS.index(algo) > ALGOS.index(current):
            state = "pending"
        elif current is not None and ALGOS.index(algo) < ALGOS.index(current):
            state = "done"
        elif current is None and train_steps:
            # all done or idle after completion
            state = "done"
        elif current is None and not train_steps and log_size > 1000:
            state = "done"  # may have failed early
    else:
        state = "pending"

    metrics_table = []
    for r in rows:
        metrics_table.append(
            {
                "step": r.get("step"),
                **{k: r.get(k) for k in METRIC_KEYS},
            }
        )

    return {
        "algo": algo,
        "state": state,
        "log_path": str(log_path),
        "log_size": log_size,
        "log_mtime": log_mtime,
        "metrics_path": str(mpath) if mpath else None,
        "n_metric_rows": len(rows),
        "n_train_steps": len(train_steps),
        "last_step": last.get("step") if last else None,
        "last": {k: (last_train or last or {}).get(k) for k in METRIC_KEYS} if (last_train or last) else {},
        "series": metrics_table,
    }


def build_status(log_lines: int = 60) -> dict[str, Any]:
    ps = _ps_snapshot()
    current = ps.get("current_algo")
    algos = [_algo_status(a, current) for a in ALGOS]
    active_log = LOG_DIR / f"{current}.log" if current else None
    # Prefer newest algo log if process detection missed
    if active_log is None or not active_log.is_file():
        newest = None
        newest_mtime = -1.0
        for a in ALGOS:
            p = LOG_DIR / f"{a}.log"
            if p.is_file() and p.stat().st_mtime > newest_mtime:
                newest_mtime = p.stat().st_mtime
                newest = p
                if current is None and ps["script_running"]:
                    current = a
        active_log = newest

    log_text = _tail_text(active_log, log_lines) if active_log else ""
    return {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log_dir": str(LOG_DIR),
        "script_running": ps["script_running"],
        "script_etime": ps["script_etime"],
        "current_algo": current,
        "train_pids": ps["train_pids"],
        "gpus": _gpu_snapshot(),
        "algos": algos,
        "active_log": str(active_log) if active_log else None,
        "log_tail": log_text,
        "recent_rewards": _recent_rewards(log_text),
    }


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AgentFlow 4-Algo Monitor</title>
<style>
  :root {
    --bg: #0f1419;
    --panel: #1a2332;
    --border: #2d3a4d;
    --text: #e7ecf3;
    --muted: #8b9bb4;
    --accent: #3d9cf0;
    --ok: #3ecf8e;
    --warn: #f0b429;
    --run: #3d9cf0;
    --pending: #5a6a82;
    --fail: #f07178;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.45;
  }
  header {
    padding: 16px 20px; border-bottom: 1px solid var(--border);
    display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: baseline;
  }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: 0.02em; }
  header .meta { color: var(--muted); font-size: 13px; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-size: 12px; font-weight: 600; text-transform: uppercase;
  }
  .badge.run { background: #1a3a5c; color: var(--run); }
  .badge.done { background: #143528; color: var(--ok); }
  .badge.pending { background: #243040; color: var(--pending); }
  .badge.idle { background: #243040; color: var(--muted); }
  main { padding: 16px 20px; display: grid; gap: 16px; }
  .grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  @media (max-width: 1100px) { .grid4 { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 640px) { .grid4 { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px;
  }
  .card h2 { margin: 0 0 8px; font-size: 14px; font-weight: 600; }
  .card.active { border-color: var(--accent); }
  .kv { display: grid; grid-template-columns: 1fr auto; gap: 2px 10px; font-size: 12px; }
  .kv span:first-child { color: var(--muted); }
  .kv span:last-child { font-variant-numeric: tabular-nums; text-align: right; }
  .split { display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }
  @media (max-width: 960px) { .split { grid-template-columns: 1fr; } }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 4px 6px; border-bottom: 1px solid var(--border); text-align: right; font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 500; }
  pre {
    margin: 0; padding: 12px; background: #0b0f14; border: 1px solid var(--border);
    border-radius: 6px; font-size: 11px; line-height: 1.35; overflow: auto;
    max-height: 420px; white-space: pre-wrap; word-break: break-word;
    font-family: "IBM Plex Mono", ui-monospace, monospace; color: #c5d0de;
  }
  .gpus { display: flex; gap: 12px; flex-wrap: wrap; }
  .gpu { flex: 1; min-width: 160px; }
  .bar { height: 6px; background: #243040; border-radius: 3px; margin-top: 4px; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: var(--accent); }
  .rewards { font-size: 12px; }
  .rewards td:nth-child(3) { color: var(--ok); font-weight: 600; }
  .err { color: var(--fail); }
</style>
</head>
<body>
<header>
  <h1>AgentFlow 4-Algo Monitor</h1>
  <span class="meta" id="hdr-meta">loading…</span>
  <span id="hdr-badge"></span>
</header>
<main>
  <section class="gpus" id="gpus"></section>
  <section class="grid4" id="algos"></section>
  <section class="split">
    <div class="card">
      <h2>Metrics (current / selected)</h2>
      <div id="metrics-wrap"></div>
    </div>
    <div class="card">
      <h2>Recent rollout rewards (from log)</h2>
      <div id="rewards-wrap"></div>
    </div>
  </section>
  <section class="card">
    <h2 id="log-title">Log tail</h2>
    <pre id="log"></pre>
  </section>
</main>
<script>
const KEYS = [
  "training/reward","val/reward","actor/pg_loss","actor/ppo_kl",
  "actor/entropy_loss","training/n_triplets","training/n_rollouts",
  "training/branch_local_count","training/n_with_anchor"
];
function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return String(v);
}
function badge(state) {
  const cls = state === "running" ? "run" : state === "done" ? "done" : "pending";
  return `<span class="badge ${cls}">${state}</span>`;
}
function render(s) {
  const run = s.script_running;
  document.getElementById("hdr-meta").textContent =
    `${s.ts_iso} · script ${run ? "UP "+(s.script_etime||"") : "DOWN"} · current=${s.current_algo||"—"}`;
  document.getElementById("hdr-badge").innerHTML = run
    ? `<span class="badge run">running</span>`
    : `<span class="badge idle">idle</span>`;

  document.getElementById("gpus").innerHTML = (s.gpus||[]).map(g => {
    const pct = g.mem_total_mb ? (100*g.mem_used_mb/g.mem_total_mb) : 0;
    return `<div class="card gpu"><h2>GPU ${g.index}</h2>
      <div class="kv">
        <span>util</span><span>${g.util.toFixed(0)}%</span>
        <span>mem</span><span>${g.mem_used_mb.toFixed(0)} / ${g.mem_total_mb.toFixed(0)} MiB</span>
        <span>temp</span><span>${g.temp_c.toFixed(0)}°C</span>
        <span>power</span><span>${g.power_w.toFixed(0)} W</span>
      </div>
      <div class="bar"><i style="width:${pct.toFixed(1)}%"></i></div>
    </div>`;
  }).join("");

  document.getElementById("algos").innerHTML = (s.algos||[]).map(a => {
    const L = a.last || {};
    const active = a.algo === s.current_algo ? " active" : "";
    return `<div class="card${active}"><h2>${a.algo} ${badge(a.state)}</h2>
      <div class="kv">
        <span>steps</span><span>${a.n_train_steps} train / ${a.n_metric_rows} rows</span>
        <span>last step</span><span>${fmt(a.last_step)}</span>
        <span>train/reward</span><span>${fmt(L["training/reward"])}</span>
        <span>val/reward</span><span>${fmt(L["val/reward"])}</span>
        <span>pg_loss</span><span>${fmt(L["actor/pg_loss"])}</span>
        <span>ppo_kl</span><span>${fmt(L["actor/ppo_kl"])}</span>
        <span>entropy</span><span>${fmt(L["actor/entropy_loss"])}</span>
        <span>n_triplets</span><span>${fmt(L["training/n_triplets"])}</span>
        <span>log size</span><span>${(a.log_size/1024).toFixed(1)} KB</span>
      </div></div>`;
  }).join("");

  const focus = (s.algos||[]).find(a => a.algo === s.current_algo) || (s.algos||[]).find(a => a.series && a.series.length) || null;
  const series = focus ? focus.series : [];
  if (series.length) {
    const head = `<tr><th>step</th>${KEYS.map(k=>`<th>${k.replace("training/","").replace("actor/","")}</th>`).join("")}</tr>`;
    const body = series.map(r => `<tr><td>${fmt(r.step)}</td>${KEYS.map(k=>`<td>${fmt(r[k])}</td>`).join("")}</tr>`).join("");
    document.getElementById("metrics-wrap").innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
  } else {
    document.getElementById("metrics-wrap").innerHTML = `<p class="meta" style="color:var(--muted)">尚无 metrics.jsonl</p>`;
  }

  const rew = s.recent_rewards || [];
  if (rew.length) {
    document.getElementById("rewards-wrap").innerHTML =
      `<table class="rewards"><thead><tr><th>src</th><th>gt</th><th>R</th><th>t(s)</th></tr></thead><tbody>` +
      rew.map(r => `<tr><td>${r.src}</td><td>${r.gt}</td><td>${fmt(r.R)}</td><td>${fmt(r.t)}</td></tr>`).join("") +
      `</tbody></table>`;
  } else {
    document.getElementById("rewards-wrap").innerHTML = `<p style="color:var(--muted);font-size:12px">等待 rollout 日志…</p>`;
  }

  document.getElementById("log-title").textContent = `Log tail — ${s.active_log || "—"}`;
  document.getElementById("log").textContent = s.log_tail || "(empty)";
}
async function tick() {
  try {
    const r = await fetch("/api/status?lines=80");
    const s = await r.json();
    render(s);
  } catch (e) {
    document.getElementById("hdr-meta").innerHTML = `<span class="err">poll failed: ${e}</span>`;
  }
}
tick();
setInterval(tick, 2000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "AgentFlow4AlgoMonitor/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quiet access log; keep errors
        if args and str(args[1]).startswith("5"):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            qs = parse_qs(parsed.query)
            lines = int(qs.get("lines", ["60"])[0])
            payload = json.dumps(build_status(log_lines=lines), ensure_ascii=False, indent=None).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/log":
            qs = parse_qs(parsed.query)
            algo = qs.get("algo", [None])[0]
            lines = int(qs.get("lines", ["100"])[0])
            path = LOG_DIR / f"{algo}.log" if algo in ALGOS else None
            text = _tail_text(path, lines) if path else ""
            self._send(200, text.encode("utf-8"), "text/plain; charset=utf-8")
            return
        self._send(404, b'{"error":"not found"}', "application/json")


def run_http(host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[monitor] http://{host}:{port}/")
    print(f"[monitor] API  http://{host}:{port}/api/status")
    print(f"[monitor] logs {LOG_DIR}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[monitor] stopped")
        httpd.server_close()


def fmt_cli(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def run_cli_plain(interval: float) -> None:
    while True:
        os.system("clear" if os.name != "nt" else "cls")
        s = build_status(40)
        print(f"=== AgentFlow 4-Algo Monitor @ {s['ts_iso']} ===")
        print(f"script={'UP ' + str(s['script_etime']) if s['script_running'] else 'DOWN'} current={s['current_algo']}")
        for a in s["algos"]:
            L = a["last"]
            print(
                f"  [{a['state']:7}] {a['algo']:10} steps={a['n_train_steps']:2} "
                f"R={fmt_cli(L.get('training/reward'))} valR={fmt_cli(L.get('val/reward'))} "
                f"kl={fmt_cli(L.get('actor/ppo_kl'))} ent={fmt_cli(L.get('actor/entropy_loss'))} "
                f"tri={fmt_cli(L.get('training/n_triplets'))}"
            )
        for g in s["gpus"]:
            print(f"  GPU{g['index']}: util={g['util']:.0f}% mem={g['mem_used_mb']:.0f}/{g['mem_total_mb']:.0f}MiB")
        print("--- log ---")
        print(s["log_tail"][-2000:])
        time.sleep(interval)


def run_cli_rich(interval: float) -> None:
    try:
        from rich.console import Console, Group
        from rich.live import Live
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich.columns import Columns
    except ImportError:
        run_cli_plain(interval)
        return

    console = Console()

    def make():
        s = build_status(40)
        head = Text.assemble(
            ("AgentFlow 4-Algo Monitor  ", "bold"),
            (s["ts_iso"] + "  ", "dim"),
            (
                f"SCRIPT UP ({s['script_etime']})  " if s["script_running"] else "SCRIPT DOWN  ",
                "bold cyan" if s["script_running"] else "bold yellow",
            ),
            (f"current={s['current_algo'] or '—'}", "green"),
        )
        t = Table(expand=True)
        for col in ("algo", "state", "steps", "train/R", "val/R", "pg_loss", "ppo_kl", "entropy", "triplets"):
            t.add_column(col, justify="right" if col not in ("algo", "state") else "left")
        for a in s["algos"]:
            L = a["last"]
            style = "cyan" if a["state"] == "running" else ("green" if a["state"] == "done" else "dim")
            t.add_row(
                a["algo"],
                a["state"],
                f"{a['n_train_steps']}/{a['n_metric_rows']}",
                fmt_cli(L.get("training/reward")),
                fmt_cli(L.get("val/reward")),
                fmt_cli(L.get("actor/pg_loss")),
                fmt_cli(L.get("actor/ppo_kl")),
                fmt_cli(L.get("actor/entropy_loss")),
                fmt_cli(L.get("training/n_triplets")),
                style=style,
            )
        gpu_panels = [
            Panel(
                f"util {g['util']:.0f}%\nmem {g['mem_used_mb']:.0f}/{g['mem_total_mb']:.0f} MiB\n"
                f"{g['temp_c']:.0f}°C  {g['power_w']:.0f}W",
                title=f"GPU {g['index']}",
                width=28,
            )
            for g in s["gpus"]
        ]
        rew = s["recent_rewards"]
        rew_txt = (
            "\n".join(f"{r['src']:8} R={r['R']:.3f}  t={r['t']:.1f}s  gt={r['gt']}" for r in rew) or "(waiting)"
        )
        parts: list[Any] = [Panel(head), t]
        if gpu_panels:
            parts.append(Columns(gpu_panels))
        parts.append(Panel(rew_txt, title="recent rewards"))
        parts.append(Panel(s["log_tail"] or "(empty)", title=s["active_log"] or "log"))
        return Group(*parts)

    with Live(make(), console=console, refresh_per_second=2, screen=True) as live:
        while True:
            time.sleep(interval)
            live.update(make())


def main() -> None:
    ap = argparse.ArgumentParser(description="Realtime monitor for agentflow_4algo_run.sh")
    ap.add_argument("--cli", action="store_true", help="terminal TUI instead of HTTP")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--interval", type=float, default=2.0, help="CLI refresh seconds")
    ap.add_argument("--once", action="store_true", help="print JSON status once and exit")
    args = ap.parse_args()

    if args.once:
        print(json.dumps(build_status(), ensure_ascii=False, indent=2))
        return
    if args.cli:
        run_cli_rich(args.interval)
        return
    run_http(args.host, args.port)


if __name__ == "__main__":
    main()
