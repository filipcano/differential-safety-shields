#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import re
import subprocess
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
SOLVERS = ["basic", "optimized", "optimized-iterative"]
PARAM_COLUMNS = [
    "k",
    "x_max",
    "y_max",
    "vx_cmd_max",
    "vy_cmd_max",
    "eta_max",
    "x_wall",
    "radii",
]


def die(message):
    raise SystemExit(message)


def require(mapping, key):
    if key not in mapping:
        die(f"missing required key: {key}")
    return mapping[key]


def resolve_path(value, base_dir):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def resolve_executable(value, base_dir):
    path = resolve_path(value, base_dir)
    if path.exists() or Path(value).expanduser().is_absolute():
        return path

    script_relative = SCRIPT_DIR / value
    if script_relative.exists():
        return script_relative
    return path


def sanitize(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def benchmark_display_id(bench):
    radii = ", ".join(str(r) for r in bench["radii"])
    return (
        f"$E^{{({bench['x_max']}, {bench['y_max']}), "
        f"({bench['vx_cmd_max']}, {bench['vy_cmd_max']}), "
        f"{bench['eta_max']}}}_{{[{radii}]}}$"
    )


def as_int(value, name):
    if not isinstance(value, int):
        die(f"{name} must be an integer")
    return value


def validate_benchmark(bench):
    for key in PARAM_COLUMNS[:-1]:
        as_int(require(bench, key), key)
    radii = require(bench, "radii")
    if not isinstance(radii, list) or len(radii) < bench["k"]:
        die("radii must be a list with at least k entries")
    if bench["k"] <= 0:
        die("k must be positive")
    if bench["x_wall"] < 0:
        die("x_wall must be nonnegative")
    return {**bench, "radii": [int(r) for r in radii[: bench["k"]]]}


def apply_derived_fields(bench, config):
    offset = bench.get("x_wall_offset", config.get("x_wall_offset"))
    if offset is not None:
        as_int(offset, "x_wall_offset")
        if "x_max" not in bench:
            die("x_wall_offset requires x_max")
        bench["x_wall"] = bench["x_max"] - offset
    return bench


def expand_benchmarks(config):
    if "benchmarks" in config:
        benchmarks = [validate_benchmark(apply_derived_fields(dict(bench), config)) for bench in config["benchmarks"]]
    else:
        base = dict(require(config, "base"))
        sweep = config.get("sweep", {})
        keys = list(sweep.keys())
        values = []
        for key in keys:
            choices = sweep[key]
            if not isinstance(choices, list) or not choices:
                die(f"sweep key {key} must be a nonempty list")
            values.append(choices)

        benchmarks = []
        for combo in itertools.product(*values):
            bench = dict(base)
            for key, value in zip(keys, combo):
                bench[key] = value
            benchmarks.append(validate_benchmark(apply_derived_fields(bench, config)))

    for i, bench in enumerate(benchmarks, start=1):
        bench.setdefault("id", f"b{i:04d}")
    return sorted(benchmarks, key=difficulty_key)


def difficulty_key(bench):
    radii = bench["radii"]
    return (
        bench["k"],
        bench["x_max"] * bench["y_max"],
        bench["x_max"],
        bench["y_max"],
        bench["x_wall"],
        bench["vx_cmd_max"] * bench["vy_cmd_max"],
        bench["vx_cmd_max"],
        bench["vy_cmd_max"],
        bench["eta_max"],
        sum(radii),
        tuple(radii),
        str(bench["id"]),
    )


def easier_or_equal(easy, hard):
    if easy["k"] > hard["k"]:
        return False

    scalar_keys = [
        "x_max",
        "y_max",
        "vx_cmd_max",
        "vy_cmd_max",
        "eta_max",
        "x_wall",
    ]
    if any(easy[key] > hard[key] for key in scalar_keys):
        return False

    return all(a <= b for a, b in zip(easy["radii"], hard["radii"]))


def benchmark_input(bench, solver, strategy_path):
    return "\n".join(
        [
            str(bench["k"]),
            solver,
            f"{bench['x_max']} {bench['y_max']}",
            f"{bench['vx_cmd_max']} {bench['vy_cmd_max']}",
            str(bench["eta_max"]),
            str(bench["x_wall"]),
            " ".join(str(r) for r in bench["radii"]),
            str(strategy_path),
            "",
        ]
    )


def parse_cost(output):
    elapsed = None
    current_rss = None
    peak_rss = None

    match = re.search(r"synthesis total cost:\s+elapsed\s+([0-9.]+)\s+s", output)
    if match:
        elapsed = float(match.group(1))

    match = re.search(r"synthesis total cost:.*?,\s+RSS\s+(unknown|[0-9.]+)", output)
    if match and match.group(1) != "unknown":
        current_rss = float(match.group(1))

    match = re.search(r"synthesis total cost:.*?,\s+peak RSS\s+(unknown|[0-9.]+)", output)
    if match and match.group(1) != "unknown":
        peak_rss = float(match.group(1))

    return elapsed, current_rss, peak_rss


def reported_pruning_orders(max_k):
    return [d for d in range(1, max_k + 1) if d != 2]


def pruning_columns(max_k):
    return [f"iter_prune_fraction_d{d}" for d in reported_pruning_orders(max_k)]


def pruning_order(column):
    match = re.fullmatch(r"iter_prune_fraction_d([0-9]+)", column)
    if not match:
        die(f"internal error: invalid pruning column {column!r}")
    return int(match.group(1))


def zero_pruning_metrics(columns, k):
    return {column: ("-" if pruning_order(column) > k else "0") for column in columns}


def format_fraction(value):
    if value == 0.0:
        return "0"
    return f"{value:.9g}"


def parse_iterative_pruning(output, max_k):
    metrics = {}
    current_order = None
    candidate_states = None
    prev_filter_rejections = 0

    def finish_stage():
        if current_order is None or current_order < 1 or current_order > max_k or current_order == 2:
            return
        candidates = 0 if candidate_states is None else candidate_states
        denominator = candidates + prev_filter_rejections
        fraction = 0.0 if denominator == 0 else prev_filter_rejections / denominator
        metrics[f"iter_prune_fraction_d{current_order}"] = format_fraction(fraction)

    for line in output.splitlines():
        match = re.search(r"=== solving stage: derivative orders <=\s+([0-9]+),", line)
        if match:
            finish_stage()
            current_order = int(match.group(1))
            candidate_states = None
            prev_filter_rejections = 0
            continue

        if current_order is None:
            continue

        match = re.search(r"reachable locally safe candidate states:\s+([0-9]+)", line)
        if match:
            candidate_states = int(match.group(1))
            continue

        match = re.search(r"previous-filter rejections:\s+([0-9]+)", line)
        if match:
            prev_filter_rejections = int(match.group(1))

    finish_stage()
    return metrics


def row_base(bench, solver, status, timeout_seconds, skip_reason=""):
    row = {
        "benchmark_id": benchmark_display_id(bench),
        "solver_mode": solver,
        "status": status,
        "timeout_seconds": timeout_seconds,
        "skip_reason": skip_reason,
    }
    for key in PARAM_COLUMNS:
        row[key] = " ".join(str(r) for r in bench[key]) if key == "radii" else bench[key]
    return row


def write_input_file(path, input_text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(input_text, encoding="utf-8")


def run_synthesis(synth_exe, input_text, timeout_seconds, log_path):
    start = time.monotonic()
    try:
        result = subprocess.run(
            [str(synth_exe)],
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        wall_seconds = time.monotonic() - start
        output = (result.stdout or "") + (result.stderr or "")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8", errors="replace")
        return result.returncode, wall_seconds, output
    except subprocess.TimeoutExpired as exc:
        wall_seconds = time.monotonic() - start
        output = ""
        if exc.stdout:
            output += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(errors="replace")
        if exc.stderr:
            output += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(errors="replace")
        output += f"\nTIMEOUT after {timeout_seconds} seconds\n"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8", errors="replace")
        return "timeout", wall_seconds, output


def flush_row(writer, file_handle, row):
    writer.writerow(row)
    file_handle.flush()


def write_issue(progress, message):
    if progress is not None:
        progress.write(message)
    else:
        print(message, flush=True)


def reduce_progress_total(progress, amount=1):
    if progress is None or progress.total is None:
        return
    progress.total = max(progress.n, progress.total - amount)
    progress.refresh()


def main():
    ap = argparse.ArgumentParser(description="Benchmark synthesis modes and stream time/memory CSV results")
    ap.add_argument("config", help="JSON benchmark config")
    ap.add_argument("--timeout-seconds", type=float, default=None, help="Override per-run timeout")
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config_dir = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))

    timeout_seconds = args.timeout_seconds
    if timeout_seconds is None:
        timeout_seconds = float(config.get("timeout_seconds", 60))
    if timeout_seconds <= 0:
        die("timeout_seconds must be positive")

    solvers = config.get("solvers", SOLVERS)
    if not isinstance(solvers, list) or not solvers:
        die("solvers must be a nonempty list")
    for solver in solvers:
        if solver not in SOLVERS:
            die(f"unknown solver mode: {solver}")

    default_output_dir = SCRIPT_DIR / "experimental_results" / "synthesis_benchmark_outputs"
    output_dir = resolve_path(config.get("output_dir", str(default_output_dir)), config_dir)
    inputs_dir = output_dir / "inputs"
    strategies_dir = output_dir / "strategies"
    logs_dir = output_dir / "logs"
    time_csv = resolve_path(config.get("time_csv", str(output_dir / "synthesis_time.csv")), config_dir)
    memory_csv = resolve_path(config.get("memory_csv", str(output_dir / "synthesis_memory.csv")), config_dir)
    synth_exe = resolve_executable(config.get("synth_exe", str(SCRIPT_DIR / "synth.o")), config_dir)

    benchmarks = expand_benchmarks(config)
    max_k = max(bench["k"] for bench in benchmarks)
    prune_fields = pruning_columns(max_k)
    total_synth_calls = len(benchmarks) * len(solvers)
    time_csv.parent.mkdir(parents=True, exist_ok=True)
    memory_csv.parent.mkdir(parents=True, exist_ok=True)

    if tqdm is None:
        die("tqdm is required for benchmark progress display; install it with: python3 -m pip install tqdm")

    time_fields = [
        "benchmark_id",
        "solver_mode",
        "status",
        "elapsed_seconds",
        "wall_seconds",
        "timeout_seconds",
        "skip_reason",
    ] + PARAM_COLUMNS + prune_fields
    memory_fields = [
        "benchmark_id",
        "solver_mode",
        "status",
        "peak_rss_mib",
        "final_rss_mib",
        "timeout_seconds",
        "skip_reason",
    ] + PARAM_COLUMNS + prune_fields

    timed_out = {solver: [] for solver in solvers}

    with time_csv.open("w", newline="", encoding="utf-8") as time_file, memory_csv.open("w", newline="", encoding="utf-8") as memory_file:
        time_writer = csv.DictWriter(time_file, fieldnames=time_fields)
        memory_writer = csv.DictWriter(memory_file, fieldnames=memory_fields)
        time_writer.writeheader()
        memory_writer.writeheader()
        time_file.flush()
        memory_file.flush()

        with tqdm(total=total_synth_calls, desc="synthesis calls", unit="call", dynamic_ncols=True) as progress:
            for bench in benchmarks:
                for solver in solvers:
                    blocker = next((old for old in timed_out[solver] if easier_or_equal(old, bench)), None)
                    base_time = row_base(bench, solver, "pending", timeout_seconds)
                    base_memory = row_base(bench, solver, "pending", timeout_seconds)
                    base_time.update(zero_pruning_metrics(prune_fields, bench["k"]))
                    base_memory.update(zero_pruning_metrics(prune_fields, bench["k"]))

                    if blocker is not None:
                        reason = f"easier benchmark {blocker['id']} timed out"
                        base_time.update({"status": "pruned_timeout", "elapsed_seconds": "", "wall_seconds": "", "skip_reason": reason})
                        base_memory.update({"status": "pruned_timeout", "peak_rss_mib": "", "final_rss_mib": "", "skip_reason": reason})
                        flush_row(time_writer, time_file, base_time)
                        flush_row(memory_writer, memory_file, base_memory)
                        reduce_progress_total(progress)
                        continue

                    run_name = f"{sanitize(bench['id'])}_{sanitize(solver)}"
                    strategy_path = strategies_dir / f"{run_name}.bin"
                    input_path = inputs_dir / f"{run_name}.txt"
                    log_path = logs_dir / f"{run_name}.log"
                    input_text = benchmark_input(bench, solver, strategy_path)
                    strategy_path.parent.mkdir(parents=True, exist_ok=True)
                    write_input_file(input_path, input_text)

                    progress.set_postfix_str(f"{bench['id']} {solver}", refresh=False)
                    returncode, wall_seconds, output = run_synthesis(synth_exe, input_text, timeout_seconds, log_path)
                    progress.update()
                    pruning_metrics = zero_pruning_metrics(prune_fields, bench["k"])
                    if solver == "optimized-iterative":
                        pruning_metrics.update(parse_iterative_pruning(output, max_k))

                    if returncode == "timeout":
                        status = "timeout"
                        elapsed = ""
                        current_rss = ""
                        peak_rss = ""
                        timed_out[solver].append(bench)
                        write_issue(progress, f"[{bench['id']} {solver}] timeout after {timeout_seconds:g}s; log: {log_path}")
                    elif returncode == 0:
                        status = "ok"
                        elapsed, current_rss, peak_rss = parse_cost(output)
                        elapsed = "" if elapsed is None else elapsed
                        current_rss = "" if current_rss is None else current_rss
                        peak_rss = "" if peak_rss is None else peak_rss
                    else:
                        status = "error"
                        elapsed, current_rss, peak_rss = parse_cost(output)
                        elapsed = "" if elapsed is None else elapsed
                        current_rss = "" if current_rss is None else current_rss
                        peak_rss = "" if peak_rss is None else peak_rss
                        write_issue(progress, f"[{bench['id']} {solver}] error exit code {returncode}; log: {log_path}")

                    base_time.update({"status": status, "elapsed_seconds": elapsed, "wall_seconds": f"{wall_seconds:.6f}", "skip_reason": ""})
                    base_memory.update({"status": status, "peak_rss_mib": peak_rss, "final_rss_mib": current_rss, "skip_reason": ""})
                    base_time.update(pruning_metrics)
                    base_memory.update(pruning_metrics)
                    flush_row(time_writer, time_file, base_time)
                    flush_row(memory_writer, memory_file, base_memory)


if __name__ == "__main__":
    main()
