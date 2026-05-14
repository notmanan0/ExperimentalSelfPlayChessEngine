from __future__ import annotations

from pathlib import Path
import json
from typing import Any


def generate_run_report(run_dir: Path) -> str:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return f"# Run Report\n\nNo summary found at {summary_path}"

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    health_path = run_dir / "replay_health.json"
    health = None
    if health_path.exists():
        health = json.loads(health_path.read_text(encoding="utf-8"))

    lines = [
        "# chessmoe Run Report",
        "",
        "## Configuration",
        f"- Hardware profile: {data.get('hardware_profile', 'unknown')}",
        f"- Quality profile: {data.get('quality_profile', 'unknown')}",
        f"- Evaluator: {data.get('evaluator', 'unknown')}",
        f"- Build type: {data.get('build_type', 'unknown')}",
        f"- GPU: {data.get('gpu', 'none')}",
        f"- Debug build: {data.get('debug_build', 'unknown')}",
        "",
        "## Generation",
        f"- Games completed: {data.get('games_completed', 0)}",
        f"- Samples written: {data.get('samples_written', 0)}",
        f"- Games/sec: {data.get('games_per_second', 0):.2f}",
        f"- Positions/sec: {data.get('positions_per_second', 0):.2f}",
        f"- Avg plies/game: {data.get('average_plies_per_game', 0):.1f}",
        f"- Elapsed: {data.get('elapsed_ms', 0) / 1000:.1f}s",
        "",
        "## Terminal Distribution",
        f"- Checkmate: {data.get('checkmate_count', 0)}",
        f"- Stalemate: {data.get('stalemate_count', 0)}",
        f"- Repetition: {data.get('repetition_count', 0)}",
        f"- Fifty-move: {data.get('fifty_move_count', 0)}",
        f"- Max plies: {data.get('max_plies_count', 0)}",
        "",
        "## Inference",
        f"- Batch fill ratio: {data.get('batch_fill_ratio', 0):.3f}",
        f"- Padding ratio: {data.get('padding_ratio', 0):.3f}",
        f"- Avg inference latency: {data.get('avg_inference_latency_ms', 0):.3f}ms",
        f"- Replay chunks: {data.get('replay_chunks', 0)}",
    ]

    if health:
        lines.extend([
            "",
            "## Replay Health",
            f"- Status: {'PASSED' if health.get('passed') else 'FAILED'}",
            f"- Total games: {health.get('total_games', 0)}",
            f"- Total samples: {health.get('total_samples', 0)}",
            f"- Average plies: {health.get('average_plies', 0):.1f}",
            f"- Draw rate: {health.get('draw_rate', 0):.3f}",
        ])
        warnings = health.get("warnings", [])
        if warnings:
            lines.append("")
            lines.append("### Warnings")
            for w in warnings:
                lines.append(f"- {w}")

    return "\n".join(lines)


def generate_html_report(run_dir: Path) -> str:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return "<html><body><h1>No run data found</h1></body></html>"

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    health_path = run_dir / "replay_health.json"
    health = None
    if health_path.exists():
        health = json.loads(health_path.read_text(encoding="utf-8"))

    profile_path = run_dir / "profile.json"
    profile = None
    if profile_path.exists():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))

    games = data.get("games_completed", 0)
    samples = data.get("samples_written", 0)
    elapsed_s = data.get("elapsed_ms", 0) / 1000
    gps = data.get("games_per_second", 0)
    pps = data.get("positions_per_second", 0)
    avg_plies = data.get("average_plies_per_game", 0)
    batch_fill = data.get("batch_fill_ratio", 0)
    padding = data.get("padding_ratio", 0)
    inf_lat = data.get("avg_inference_latency_ms", 0)

    checkmate = data.get("checkmate_count", 0)
    stalemate = data.get("stalemate_count", 0)
    repetition = data.get("repetition_count", 0)
    fifty_move = data.get("fifty_move_count", 0)
    max_plies = data.get("max_plies_count", 0)
    terminal_total = checkmate + stalemate + repetition + fifty_move + max_plies

    health_status = "N/A"
    health_class = "neutral"
    if health:
        health_status = "PASSED" if health.get("passed") else "FAILED"
        health_class = "pass" if health.get("passed") else "fail"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>chessmoe Run Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2rem; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; }}
  h2 {{ color: #79c0ff; margin-top: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
  .card {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 1rem; }}
  .card .label {{ color: #8b949e; font-size: 0.85rem; text-transform: uppercase; }}
  .card .value {{ font-size: 1.5rem; font-weight: 600; color: #f0f6fc; margin-top: 0.25rem; }}
  .pass {{ color: #3fb950; }}
  .fail {{ color: #f85149; }}
  .neutral {{ color: #8b949e; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 0.5rem; }}
  th, td {{ text-align: left; padding: 0.5rem; border-bottom: 1px solid #21262d; }}
  th {{ color: #8b949e; font-weight: 500; }}
  .bar {{ height: 20px; background: #238636; border-radius: 3px; }}
  .warn {{ color: #d29922; }}
</style>
</head>
<body>
<h1>chessmoe Run Report</h1>

<div class="grid">
  <div class="card"><div class="label">Games</div><div class="value">{games}</div></div>
  <div class="card"><div class="label">Samples</div><div class="value">{samples}</div></div>
  <div class="card"><div class="label">Games/sec</div><div class="value">{gps:.2f}</div></div>
  <div class="card"><div class="label">Positions/sec</div><div class="value">{pps:.0f}</div></div>
  <div class="card"><div class="label">Avg plies</div><div class="value">{avg_plies:.1f}</div></div>
  <div class="card"><div class="label">Elapsed</div><div class="value">{elapsed_s:.1f}s</div></div>
  <div class="card"><div class="label">Batch fill</div><div class="value">{batch_fill:.1%}</div></div>
  <div class="card"><div class="label">Health</div><div class="value {health_class}">{health_status}</div></div>
</div>

<h2>Configuration</h2>
<table>
  <tr><th>Key</th><th>Value</th></tr>
  <tr><td>Hardware profile</td><td>{data.get('hardware_profile', 'unknown')}</td></tr>
  <tr><td>Quality profile</td><td>{data.get('quality_profile', 'unknown')}</td></tr>
  <tr><td>Evaluator</td><td>{data.get('evaluator', 'unknown')}</td></tr>
  <tr><td>Build type</td><td>{data.get('build_type', 'unknown')}</td></tr>
  <tr><td>GPU</td><td>{data.get('gpu', 'none')}</td></tr>
</table>

<h2>Terminal Distribution</h2>
<table>
  <tr><th>Reason</th><th>Count</th><th>Percentage</th></tr>
  <tr><td>Checkmate</td><td>{checkmate}</td><td>{checkmate / max(1, terminal_total) * 100:.1f}%</td></tr>
  <tr><td>Stalemate</td><td>{stalemate}</td><td>{stalemate / max(1, terminal_total) * 100:.1f}%</td></tr>
  <tr><td>Repetition</td><td>{repetition}</td><td>{repetition / max(1, terminal_total) * 100:.1f}%</td></tr>
  <tr><td>Fifty-move</td><td>{fifty_move}</td><td>{fifty_move / max(1, terminal_total) * 100:.1f}%</td></tr>
  <tr><td>Max plies</td><td>{max_plies}</td><td>{max_plies / max(1, terminal_total) * 100:.1f}%</td></tr>
</table>

<h2>Inference</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Batch fill ratio</td><td>{batch_fill:.3f}</td></tr>
  <tr><td>Padding ratio</td><td>{padding:.3f}</td></tr>
  <tr><td>Avg inference latency</td><td>{inf_lat:.3f}ms</td></tr>
  <tr><td>Replay chunks</td><td>{data.get('replay_chunks', 0)}</td></tr>
</table>
"""

    if health:
        html += f"""
<h2>Replay Health</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total games</td><td>{health.get('total_games', 0)}</td></tr>
  <tr><td>Total samples</td><td>{health.get('total_samples', 0)}</td></tr>
  <tr><td>Average plies</td><td>{health.get('average_plies', 0):.1f}</td></tr>
  <tr><td>Draw rate</td><td>{health.get('draw_rate', 0):.3f}</td></tr>
</table>
"""
        warnings = health.get("warnings", [])
        if warnings:
            html += "<h3 class='warn'>Warnings</h3><ul>"
            for w in warnings:
                html += f"<li class='warn'>{w}</li>"
            html += "</ul>"

    if profile:
        html += "\n<h2>Profile Breakdown</h2>\n<table>\n"
        html += "<tr><th>Metric</th><th>Value</th></tr>\n"
        for k, v in profile.items():
            html += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
        html += "</table>\n"

    html += "</body></html>"
    return html


def generate_registry_report(registry_path: Path) -> str:
    from chessmoe.models.registry import ModelRegistry

    registry = ModelRegistry(registry_path)
    return registry.format_registry()
