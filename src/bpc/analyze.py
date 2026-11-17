"""Score each method and its paired difference from the configured reference."""

import csv
import math
import statistics
from pathlib import Path
from statistics import NormalDist

from .io import atomic_path, write_json, write_text
from .loader import episode_path, read_episode


def _wilson(successes, trials, confidence):
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(
        rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _rows(config, run_dir, method, required):
    evaluation = config["evaluation"]
    rows = []
    for index in range(int(evaluation["init_index_start"]),
                       int(evaluation["init_index_stop"])):
        path = episode_path(Path(run_dir) / "episodes" / method / f"{index:03d}.json")
        if not path.is_file():
            if required:
                raise RuntimeError(f"{method} is incomplete: missing init {index}")
            return None
        row = read_episode(path)
        if row.get("method") != method or row.get("init_index") != index:
            raise ValueError(f"{path} contains the wrong episode")
        rows.append(row)
    return rows


def _score(rows, confidence):
    successes = sum(row["success"] for row in rows)
    steps = [row["steps_to_success"] for row in rows if row["success"]]
    return {
        "episodes": len(rows),
        "successes": successes,
        "failures": len(rows) - successes,
        "success_rate": successes / len(rows),
        "wilson_interval": list(_wilson(successes, len(rows), confidence)),
        "median_steps_to_success": statistics.median(steps) if steps else None,
        "mean_episode_seconds": statistics.fmean(row["seconds"] for row in rows),
    }


def _mcnemar(reference, contender):
    paired = list(zip(reference, contender))
    ref_only = sum(a["success"] and not b["success"] for a, b in paired)
    contender_only = sum(b["success"] and not a["success"] for a, b in paired)
    discordant = ref_only + contender_only
    tail = sum(math.comb(discordant, k) for k in range(min(ref_only, contender_only) + 1))
    p_value = min(1.0, 2 * tail / (2 ** discordant)) if discordant else 1.0
    return {
        "both_success": sum(a["success"] and b["success"] for a, b in paired),
        "reference_only": ref_only,
        "contender_only": contender_only,
        "both_failure": sum(not a["success"] and not b["success"] for a, b in paired),
        "success_rate_difference": (
            sum(row["success"] for row in contender)
            - sum(row["success"] for row in reference)) / len(paired),
        "exact_mcnemar_p": p_value,
    }


def _plot(scores, output, title):
    names = list(scores)
    width, height = 1100, 520
    left, right, top, bottom = 75, 30, 55, 145
    plot_height = height - top - bottom
    slot = (width - left - right) / len(names)
    y = lambda value: top + (1 - value) * plot_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="75" y="30" font-family="sans-serif" font-size="20">{title} method outcomes</text>',
    ]
    for tick in range(5):
        value = tick / 4
        yy = y(value)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.0%}</text>')
    for index, name in enumerate(names):
        score = scores[name]
        center = left + slot * (index + 0.5)
        bar_width = min(90, slot * 0.55)
        rate = score["success_rate"]
        low, high = score["wilson_interval"]
        lines.extend([
            f'<rect x="{center-bar_width/2:.1f}" y="{y(rate):.1f}" width="{bar_width:.1f}" height="{y(0)-y(rate):.1f}" fill="#2563eb"/>',
            f'<line x1="{center:.1f}" y1="{y(high):.1f}" x2="{center:.1f}" y2="{y(low):.1f}" stroke="#111827" stroke-width="2"/>',
            f'<line x1="{center-8:.1f}" y1="{y(high):.1f}" x2="{center+8:.1f}" y2="{y(high):.1f}" stroke="#111827" stroke-width="2"/>',
            f'<line x1="{center-8:.1f}" y1="{y(low):.1f}" x2="{center+8:.1f}" y2="{y(low):.1f}" stroke="#111827" stroke-width="2"/>',
            f'<text x="{center:.1f}" y="{y(rate)-8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{score["successes"]}/{score["episodes"]}</text>',
            f'<text transform="translate({center+4:.1f} {height-bottom+18}) rotate(35)" text-anchor="start" font-family="sans-serif" font-size="12">{name}</text>',
        ])
    lines.append('</svg>\n')
    write_text(output, "\n".join(lines))


def analyze(config, run_dir, requested=None):
    run_dir = Path(run_dir)
    evaluation = config["evaluation"]
    task = evaluation["task"]
    selected = [requested] if requested else config["methods"]
    for method in selected:
        _rows(config, run_dir, method, required=True)
    complete = {
        method: rows for method in config["methods"]
        if (rows := _rows(config, run_dir, method, required=False)) is not None
    }
    confidence = float(config["analysis"]["confidence"])
    scores = {method: _score(rows, confidence) for method, rows in complete.items()}
    reference = config["analysis"]["reference_method"]
    comparisons = {}
    if reference in complete:
        comparisons = {
            method: _mcnemar(complete[reference], rows)
            for method, rows in complete.items() if method != reference
        }
    summary = {
        "experiment": config["experiment"]["name"],
        "benchmark": evaluation["benchmark"],
        "task": task,
        "confidence": confidence,
        "reference_method": reference,
        "methods": scores,
        "paired_comparisons": comparisons,
    }
    write_json(run_dir / "summary.json", summary)

    fields = ["method", "init_index", "seed", "success", "steps_to_success",
              "controller_steps", "warmup_demo", "seconds", "trace"]
    with atomic_path(run_dir / "results.csv") as temporary:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for method, rows in complete.items():
                writer.writerows({key: method if key == "method" else row.get(key)
                                  for key in fields} for row in rows)
    _plot(scores, run_dir / "outcomes.svg", summary["task"])
    table = [f"| method | successes | rate | {confidence:.0%} Wilson |",
             "|---|---:|---:|---:|"]
    for method, score in scores.items():
        low, high = score["wilson_interval"]
        table.append(f"| {method} | {score['successes']}/{score['episodes']} | "
                     f"{score['success_rate']:.1%} | [{low:.1%}, {high:.1%}] |")
    write_text(run_dir / "report.md", (
        f"# {config['experiment']['name']}\n\n" + "\n".join(table) +
        "\n\n`summary.json` contains the paired exact McNemar comparisons; "
        "`results.csv` contains every completed endpoint episode.\n"))
    print("[analyze] " + ", ".join(
        f"{name}={score['successes']}/{score['episodes']}" for name, score in scores.items()),
        flush=True)
    return summary
