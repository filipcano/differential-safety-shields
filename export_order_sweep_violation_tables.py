#!/usr/bin/env python3
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIGS = [
    SCRIPT_DIR / "experimental_setups" / "order_sweep_wall_config.json",
    SCRIPT_DIR / "experimental_setups" / "order_sweep_obstacle_config.json",
]
DEFAULT_OUTPUT = SCRIPT_DIR / "figures_paper" / "order_sweep_violation_tables.tex"
TABLE_COLUMN_SPACING_SCALE = "1.2"


def die(message):
    raise SystemExit(message)


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def resolve_path(value, base_dir):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def require(config, key):
    if key not in config:
        die(f"missing required config key {key!r}")
    return config[key]


def load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config_path, config


def joined_csv_path(config_path, config):
    config_dir = config_path.parent
    default_output_dir = SCRIPT_DIR / "experimental_results" / "order_sweep_outputs"
    output_dir = resolve_path(config.get("output_dir", str(default_output_dir)), config_dir)
    return resolve_path(config.get("joined_csv", str(output_dir / "traces_by_order.csv")), config_dir)


def run_order_sweep(config_path, resynthesize):
    cmd = [sys.executable, str(SCRIPT_DIR / "run_order_sweep.py"), str(config_path)]
    if resynthesize:
        cmd.append("--resynthesize")
    subprocess.run(cmd, check=True)


def read_trace_rows(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            die(f"{csv_path} is empty or has no CSV header")
        required = ["order", "trace", "step", "x", "y"]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            die(f"{csv_path} is missing required columns: {', '.join(missing)}")
        return list(reader)


def to_int(value, column, csv_path):
    try:
        return int(value)
    except ValueError:
        die(f"invalid integer in {csv_path}, column {column!r}: {value!r}")


def trace_groups(rows, csv_path):
    groups = {}
    for row in rows:
        shield_order = to_int(row["order"], "order", csv_path)
        trace_key = row.get("trace_in_order", row["trace"])
        trace_id = to_int(trace_key, "trace_in_order" if "trace_in_order" in row else "trace", csv_path)
        groups.setdefault(shield_order, {}).setdefault(trace_id, []).append(row)

    for by_trace in groups.values():
        for trace_rows in by_trace.values():
            trace_rows.sort(key=lambda row: to_int(row["step"], "step", csv_path))
    return groups


def is_transition_row(row):
    return (row.get("reason") or "").strip() != "no_allowed_action"


def derivative_at_next(next_pos, history, order):
    dx = next_pos[0]
    dy = next_pos[1]
    for i in range(1, order + 1):
        coeff = math.comb(order, i)
        if i % 2:
            coeff = -coeff
        prev = history[i - 1]
        dx += coeff * prev[0]
        dy += coeff * prev[1]
    return dx, dy


def violation_counts_for_trace(trace_rows, config, max_order, csv_path):
    x_init = int(config.get("x_init", 0))
    y_init = int(config.get("y_init", 0))
    radii = require(config, "radii")
    history = [(x_init, y_init) for _ in range(max_order)]
    counts = {order: 0 for order in range(1, max_order + 1)}

    for row in trace_rows:
        if not is_transition_row(row):
            continue

        next_pos = (
            to_int(row["x"], "x", csv_path),
            to_int(row["y"], "y", csv_path),
        )
        for order in range(1, max_order + 1):
            dx, dy = derivative_at_next(next_pos, history, order)
            radius = int(radii[order - 1])
            if dx * dx + dy * dy > radius * radius:
                counts[order] += 1

        history = [next_pos] + history[:-1]

    return counts


def analyze_config(config_path, config):
    max_order = int(require(config, "k"))
    radii = require(config, "radii")
    if not isinstance(radii, list) or len(radii) < max_order:
        die(f"{config_path}: radii must contain at least k entries")

    csv_path = joined_csv_path(config_path, config)
    if not csv_path.exists():
        die(f"joined trace CSV does not exist: {csv_path}")

    groups = trace_groups(read_trace_rows(csv_path), csv_path)
    matrix = {}
    for shield_order in range(1, max_order + 1):
        by_trace = groups.get(shield_order, {})
        if not by_trace:
            matrix[shield_order] = {order: None for order in range(1, max_order + 1)}
            continue

        totals = {order: 0 for order in range(1, max_order + 1)}
        for trace_rows in by_trace.values():
            counts = violation_counts_for_trace(trace_rows, config, max_order, csv_path)
            for order, count in counts.items():
                totals[order] += count

        trace_count = len(by_trace)
        matrix[shield_order] = {
            order: totals[order] / trace_count
            for order in range(1, max_order + 1)
        }
    return csv_path, matrix


def scenario_title(config_path):
    stem = config_path.stem.lower()
    if "obstacle" in stem:
        return "Obstacle"
    if "wall" in stem:
        return "Wall"
    return config_path.stem.replace("_", " ").title()


def scenario_label(config_path):
    stem = config_path.stem.lower().replace("_config", "")
    return "tab:" + stem.replace("_", "-") + "-violations"


def combined_label(config_paths):
    labels = []
    for config_path in config_paths:
        title = scenario_title(config_path).lower()
        labels.append(title.replace(" ", "-"))
    return "tab:" + "-".join(labels) + "-order-sweep-violations"


def format_average(value):
    if value is None:
        return "--"
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.2f}"


def render_table(config_path, config, matrix):
    max_order = int(require(config, "k"))
    title = scenario_title(config_path)
    column_spec = "l" + ("r" * max_order)

    lines = []
    lines.append("\\begin{table}")
    lines.append("\\centering")
    lines.append(f"\\setlength{{\\tabcolsep}}{{{TABLE_COLUMN_SPACING_SCALE}\\tabcolsep}}")
    lines.append(
        "\\caption{Average number of derivative-safety violations per trace "
        + f"for the {latex_escape(title.lower())} order sweep."
        + "}"
    )
    lines.append(f"\\label{{{latex_escape(scenario_label(config_path))}}}")
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\toprule")
    lines.append(
        "Shield order & "
        f"\\multicolumn{{{max_order}}}{{c}}{{Constraint order}}\\\\"
    )
    lines.append(f"\\cline{{2-{max_order + 1}}}")
    lines.append(" & " + " & ".join(f"$d={order}$" for order in range(1, max_order + 1)) + r" \\")
    lines.append("\\midrule")
    for shield_order in range(1, max_order + 1):
        cells = [f"$k={shield_order}$"]
        cells.extend(format_average(matrix[shield_order][order]) for order in range(1, max_order + 1))
        lines.append(" & ".join(cells) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def render_combined_table(results):
    if not results:
        return ""

    max_order = int(require(results[0][1], "k"))
    if any(int(require(config, "k")) != max_order for _path, config, _matrix in results):
        die("combined violation table requires all configs to have the same k")

    titles = [scenario_title(config_path) for config_path, _config, _matrix in results]
    column_spec = "l" + ("r" * max_order)
    for _ in results[1:]:
        column_spec += "|" + ("r" * max_order)

    lines = []
    lines.append("\\begin{table}")
    lines.append("\\centering")
    lines.append(f"\\setlength{{\\tabcolsep}}{{{TABLE_COLUMN_SPACING_SCALE}\\tabcolsep}}")
    lines.append("\\caption{Average number of derivative-safety violations per trace.}")
    lines.append(f"\\label{{{latex_escape(combined_label([path for path, _config, _matrix in results]))}}}")
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\toprule")
    header = ["\\multirow{2}{*}{Shield order}"]
    for title in titles:
        header.append(f"\\multicolumn{{{max_order}}}{{c}}{{{latex_escape(title)}}}")
    lines.append(" & ".join(header) + r" \\")

    clines = []
    first = 2
    for _ in results:
        last = first + max_order - 1
        clines.append(f"\\cline{{{first}-{last}}}")
        first = last + 1
    lines.append(" ".join(clines))

    subheader = [" "]
    for _ in results:
        subheader.extend(f"$d={order}$" for order in range(1, max_order + 1))
    lines.append(" & ".join(subheader) + r" \\")
    lines.append("\\midrule")

    for shield_order in range(1, max_order + 1):
        cells = [f"$k={shield_order}$"]
        for _config_path, _config, matrix in results:
            cells.extend(format_average(matrix[shield_order][order]) for order in range(1, max_order + 1))
        lines.append(" & ".join(cells) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Run order sweeps and export derivative-safety violation tables."
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[str(path) for path in DEFAULT_CONFIGS],
        help="Order-sweep JSON configs to run and analyze.",
    )
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output .tex file.",
    )
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="Analyze existing joined trace CSVs without running run_order_sweep.py.",
    )
    ap.add_argument(
        "--resynthesize",
        action="store_true",
        help="Pass --resynthesize to run_order_sweep.py.",
    )
    args = ap.parse_args()

    results = []
    for config_arg in args.configs:
        config_path, config = load_config(config_arg)
        if not args.skip_run:
            run_order_sweep(config_path, args.resynthesize)
        _csv_path, matrix = analyze_config(config_path, config)
        results.append((config_path, config, matrix))

    rendered_tables = ["% Requires: \\usepackage{booktabs,multirow}"]
    if len(results) == 2:
        rendered_tables.extend(["", render_combined_table(results)])
    else:
        for config_path, config, matrix in results:
            rendered_tables.extend(["", render_table(config_path, config, matrix)])

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (SCRIPT_DIR / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rendered_tables) + "\n", encoding="utf-8")
    print(f"wrote violation tables to: {output_path}")


if __name__ == "__main__":
    main()
