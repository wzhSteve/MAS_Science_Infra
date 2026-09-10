"""Single-file Science MAS HTML dashboard (works without AGL on :4747)."""

from __future__ import annotations

import html
import json
from typing import Any, Dict, Optional, Sequence

from science_infra.ui.dashboard import build_dashboard_model, svg_reward_chart


def render_status_html(
    batch: Any,
    hypotheses: Sequence[Any],
    *,
    dashboard: str = "",
    train_signal: Optional[Dict[str, Any]] = None,
) -> str:
    model = build_dashboard_model(batch, hypotheses, train_signal=train_signal, dashboard=dashboard)
    return render_dashboard_html(model)


def render_dashboard_html(model: Dict[str, Any]) -> str:
    traj_rows = []
    for i, row in enumerate(model.get("trajectories") or []):
        rid = html.escape(str(row.get("trajectory_id") or "")[:16])
        ans = html.escape(str(row.get("answer") or "")[:80])
        kinds = html.escape(",".join(str(k) for k in (row.get("kinds") or [])[:14]))
        fmt = "yes" if row.get("format_ok") else "no"
        rew = row.get("reward")
        rew_s = "" if rew is None else f"{float(rew):.3f}"
        traj_rows.append(
            "<tr data-idx="
            f'"{i}"><td>{i + 1}</td><td>{rid}</td><td>{html.escape(rew_s)}</td>'
            f"<td>{fmt}</td><td>{int(row.get('n_search') or 0)}</td>"
            f"<td>{int(row.get('n_python') or 0)}</td><td>{ans}</td><td>{kinds}</td></tr>"
        )
    hyp_rows = []
    plugins = []
    for h in model.get("hypotheses") or []:
        plugin = str(h.get("plugin") or "")
        if plugin and plugin not in plugins:
            plugins.append(plugin)
        hyp_rows.append(
            "<tr data-plugin="
            f'"{html.escape(plugin)}"><td>{html.escape(plugin)}</td>'
            f"<td>{html.escape(str(h.get('event_id') or ''))}</td>"
            f"<td>{html.escape(str(h.get('message') or ''))}</td></tr>"
        )
    plugin_opts = "".join(f'<option value="{html.escape(p)}">{html.escape(p)}</option>' for p in plugins)
    counts = model.get("event_counts") or {}
    count_chips = "".join(
        f'<span class="chip">{html.escape(str(k))} · {int(v)}</span>' for k, v in sorted(counts.items())
    ) or '<span class="muted">(none)</span>'
    mean = model.get("mean_reward")
    mean_s = "—" if mean is None else f"{float(mean):.4f}"
    dash_raw = str(model.get("dashboard") or "")
    dash = html.escape(dash_raw) if dash_raw else "(none)"
    dash_href = html.escape(dash_raw) if dash_raw else "#"
    ts = model.get("train_signal") or {}
    adv = html.escape(str(ts.get("advantage") or "(unset)"))
    loss = html.escape(str(ts.get("loss") or "(unset)"))
    payload = json.dumps(model, ensure_ascii=False).replace("<", "\\u003c")
    chart = svg_reward_chart(model.get("rewards") or [])
    empty_traj = '<tr><td colspan="8" class="muted">(empty)</td></tr>'
    empty_hyp = '<tr><td colspan="3" class="muted">(no hypotheses)</td></tr>'
    return (
        "<!DOCTYPE html>\n"
        '<html lang="zh">\n'
        "<head>\n"
        '  <meta charset="utf-8"/>\n'
        "  <title>science-infra status</title>\n"
        "  <style>\n"
        "    :root { --bg:#f4f6fb; --card:#fff; --line:#d8dee9; --muted:#6c757d; --accent:#228be6; }\n"
        "    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: var(--bg); color: #1a1b1e; }\n"
        "    header { padding: 1.25rem 1.5rem 0.5rem; }\n"
        "    h1 { margin: 0 0 .25rem; font-size: 1.45rem; }\n"
        "    .wrap { padding: 0 1.5rem 2rem; display: grid; gap: 1rem; }\n"
        "    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .75rem; }\n"
        "    .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 1rem; }\n"
        "    .card h2 { margin: 0 0 .75rem; font-size: 1rem; }\n"
        "    .stat .label { color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; }\n"
        "    .stat .value { font-size: 1.35rem; font-weight: 650; margin-top: .2rem; font-variant-numeric: tabular-nums; }\n"
        "    table { border-collapse: collapse; width: 100%; font-size: .9rem; }\n"
        "    td, th { border-bottom: 1px solid var(--line); padding: .4rem .55rem; text-align: left; vertical-align: top; }\n"
        "    tbody tr { cursor: pointer; }\n"
        "    tbody tr:hover, tbody tr.active { background: #e7f5ff; }\n"
        "    .muted { color: var(--muted); }\n"
        "    .chip { display: inline-block; background: #eef3f9; border-radius: 999px; padding: .15rem .55rem; margin: .15rem .25rem 0 0; font-size: .8rem; }\n"
        "    a { color: var(--accent); }\n"
        "    #event-panel { white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: .8rem; max-height: 280px; overflow: auto; }\n"
        "    .toolbar { display: flex; gap: .75rem; align-items: center; margin-bottom: .5rem; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <header>\n"
        "    <h1>science-infra status</h1>\n"
        '    <p class="muted">Science MAS dashboard — trajectories, harness, TrainSignal. Open the same JSON in Agent-lightning Dashboard → Science MAS.</p>\n'
        "  </header>\n"
        '  <div class="wrap">\n'
        '    <div class="stats">\n'
        f'      <div class="card stat"><div class="label">mean_reward</div><div class="value">{html.escape(mean_s)}</div></div>\n'
        f'      <div class="card stat"><div class="label">trajectories</div><div class="value">{int(model.get("n") or 0)}</div></div>\n'
        f'      <div class="card stat"><div class="label">errors</div><div class="value">{int(model.get("n_error") or 0)}</div></div>\n'
        f'      <div class="card stat"><div class="label">feedback hops</div><div class="value">{int(model.get("n_feedback") or 0)}</div></div>\n'
        f'      <div class="card stat"><div class="label">harness hits</div><div class="value">{int(model.get("n_hypotheses") or 0)}</div></div>\n'
        "    </div>\n"
        '    <div class="card" id="train-signal">\n'
        "      <h2>TrainSignal / dashboard</h2>\n"
        f"      <p>advantage=<strong>{adv}</strong> · loss=<strong>{loss}</strong></p>\n"
        f'      <p>dashboard: <a href="{dash_href}">{dash}</a></p>\n'
        f'      <div>{count_chips}</div>\n'
        "    </div>\n"
        '    <div class="card">\n'
        "      <h2>Episode reward</h2>\n"
        f"      {chart}\n"
        "    </div>\n"
        '    <div class="card">\n'
        "      <h2>trajectories</h2>\n"
        '      <p class="muted">Click a row to inspect events.</p>\n'
        "      <table>\n"
        "        <thead><tr><th>#</th><th>id</th><th>reward</th><th>format</th><th>search</th><th>python</th><th>answer</th><th>events</th></tr></thead>\n"
        f"        <tbody id=\"traj-body\">{''.join(traj_rows) or empty_traj}</tbody>\n"
        "      </table>\n"
        '      <h2 style="margin-top:1rem">events</h2>\n'
        '      <pre id="event-panel" class="muted">(select a trajectory)</pre>\n'
        "    </div>\n"
        '    <div class="card" id="harness">\n'
        "      <h2>harness</h2>\n"
        '      <div class="toolbar">\n'
        '        <label>plugin <select id="hyp-filter"><option value="">all</option>'
        f"{plugin_opts}</select></label>\n"
        "      </div>\n"
        "      <table>\n"
        "        <thead><tr><th>plugin</th><th>event_id</th><th>message</th></tr></thead>\n"
        f"        <tbody id=\"hyp-body\">{''.join(hyp_rows) or empty_hyp}</tbody>\n"
        "      </table>\n"
        "    </div>\n"
        "  </div>\n"
        f'  <script id="science-data" type="application/json">{payload}</script>\n'
        "  <script>\n"
        "    const DATA = JSON.parse(document.getElementById('science-data').textContent);\n"
        "    const panel = document.getElementById('event-panel');\n"
        "    document.getElementById('traj-body').addEventListener('click', (ev) => {\n"
        "      const tr = ev.target.closest('tr[data-idx]');\n"
        "      if (!tr) return;\n"
        "      document.querySelectorAll('#traj-body tr').forEach((r) => r.classList.remove('active'));\n"
        "      tr.classList.add('active');\n"
        "      const row = (DATA.trajectories || [])[Number(tr.dataset.idx)];\n"
        "      panel.classList.remove('muted');\n"
        "      panel.textContent = row ? JSON.stringify(row.events || [], null, 2) : '(empty)';\n"
        "    });\n"
        "    const filt = document.getElementById('hyp-filter');\n"
        "    filt.addEventListener('change', () => {\n"
        "      const v = filt.value;\n"
        "      document.querySelectorAll('#hyp-body tr[data-plugin]').forEach((r) => {\n"
        "        r.style.display = (!v || r.dataset.plugin === v) ? '' : 'none';\n"
        "      });\n"
        "    });\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )
