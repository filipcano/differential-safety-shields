#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def die(message):
    raise SystemExit(message)


def require(config, key):
    if key not in config:
        die(f"missing required config key: {key}")
    return config[key]


def as_int(config, key):
    value = require(config, key)
    if not isinstance(value, int):
        die(f"config key {key} must be an integer")
    return value


def optional_int(config, key, default):
    value = config.get(key, default)
    if not isinstance(value, int):
        die(f"config key {key} must be an integer")
    return value


def resolve_path(value, base_dir):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def policy_tokens(config):
    if "policy" not in config or config["policy"] is None:
        return []

    policy = config["policy"]
    if isinstance(policy, str):
        return policy.split()
    if isinstance(policy, list):
        return [str(token) for token in policy]
    if isinstance(policy, dict):
        kind = str(require(policy, "kind"))
        if kind == "uniform":
            return ["uniform"]

        mode = str(require(policy, "mode"))
        tokens = [kind, mode]
        if kind == "x_axis_max":
            if mode == "softmax":
                tokens.append(str(require(policy, "temperature")))
            return tokens
        if kind == "x_target_speed":
            tokens.append(str(require(policy, "target_speed")))
            if mode == "softmax":
                tokens.append(str(require(policy, "temperature")))
            return tokens
        die(f"unknown policy kind in config: {kind}")

    die("config key policy must be a string, list, object, or null")
    return []


def obstacle_tokens(config):
    obstacles = config.get("obstacles", [])
    if obstacles is None:
        obstacles = []
    if not isinstance(obstacles, list):
        die("config key obstacles must be a list of [x_min, y_min, x_max, y_max] rectangles")

    tokens = ["obstacles", str(len(obstacles))]
    for i, rect in enumerate(obstacles):
        if not isinstance(rect, list) or len(rect) != 4:
            die(f"obstacles[{i}] must be [x_min, y_min, x_max, y_max]")
        values = []
        for value in rect:
            if not isinstance(value, int):
                die(f"obstacles[{i}] coordinates must be integers")
            values.append(value)
        tokens.extend(str(value) for value in values)
    return tokens if obstacles else []


def write_text_file(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8")


def write_synthesis_input(path, config, order, strategy_path):
    radii = require(config, "radii")[:order]
    obstacles = obstacle_tokens(config)
    lines = [
        order,
        require(config, "solver_mode"),
        f"{require(config, 'x_max')} {require(config, 'y_max')}",
        f"{require(config, 'vx_cmd_max')} {require(config, 'vy_cmd_max')}",
        require(config, "eta_max"),
        require(config, "x_wall"),
        " ".join(str(r) for r in radii),
        strategy_path,
    ]
    if obstacles:
        lines.append(" ".join(obstacles))
    write_text_file(path, lines)


def write_simulation_input(path, config, order, strategy_path, trace_path, seed, policy):
    radii = require(config, "radii")[:order]
    obstacles = obstacle_tokens(config)
    lines = [
        order,
        f"{optional_int(config, 'x_init', 0)} {optional_int(config, 'y_init', 0)}",
        f"{require(config, 'x_max')} {require(config, 'y_max')}",
        f"{require(config, 'vx_cmd_max')} {require(config, 'vy_cmd_max')}",
        require(config, "eta_max"),
        require(config, "x_wall"),
        " ".join(str(r) for r in radii),
        strategy_path,
        require(config, "num_traces_per_order"),
        require(config, "max_steps"),
        trace_path,
        seed,
    ]
    if policy:
        lines.append(" ".join(policy))
    if obstacles:
        lines.append(" ".join(obstacles))
    write_text_file(path, lines)


def log_tail(path, limit=40):
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def run_with_input(executable, input_path, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as stdin, log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([str(executable)], stdin=stdin, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        die(f"{executable} failed for {input_path}\n\nLast log lines:\n{log_tail(log_path)}")


def validate_config(config):
    k = as_int(config, "k")
    if k <= 0:
        die("config key k must be positive")

    radii = require(config, "radii")
    if not isinstance(radii, list) or len(radii) < k:
        die("config key radii must be a list with at least k entries")

    for key in [
        "x_max",
        "y_max",
        "vx_cmd_max",
        "vy_cmd_max",
        "eta_max",
        "x_wall",
        "num_traces_per_order",
        "max_steps",
        "random_seed",
    ]:
        as_int(config, key)

    if as_int(config, "num_traces_per_order") <= 0:
        die("config key num_traces_per_order must be positive")
    if as_int(config, "max_steps") <= 0:
        die("config key max_steps must be positive")

    x_init = optional_int(config, "x_init", 0)
    y_init = optional_int(config, "y_init", 0)
    if x_init < 0 or x_init > config["x_max"] or y_init < 0 or y_init > config["y_max"]:
        die("initial position is outside the domain")
    if x_init > config["x_wall"]:
        die("initial position violates x_wall")

    for i, rect in enumerate(config.get("obstacles") or []):
        if not isinstance(rect, list) or len(rect) != 4:
            die(f"obstacles[{i}] must be [x_min, y_min, x_max, y_max]")
        if not all(isinstance(value, int) for value in rect):
            die(f"obstacles[{i}] coordinates must be integers")
        x_min, y_min, x_max, y_max = rect
        if x_min < 0 or y_min < 0 or x_max > config["x_max"] or y_max > config["y_max"] or x_min > x_max or y_min > y_max:
            die(f"obstacles[{i}] is outside the domain or has invalid bounds")
        if x_init >= x_min and x_init <= x_max and y_init >= y_min and y_init <= y_max:
            die(f"initial position is inside obstacles[{i}]")


def join_traces(trace_paths, joined_csv, num_traces_per_order):
    frames = []
    next_trace_id = 0
    for order, trace_path in trace_paths:
        df = pd.read_csv(trace_path)
        if "trace" not in df.columns:
            die(f"trace CSV has no trace column: {trace_path}")

        original_trace = df["trace"].copy()
        df["trace"] = original_trace + next_trace_id
        df.insert(1, "order", order)
        df.insert(2, "trace_in_order", original_trace)
        frames.append(df)
        next_trace_id += num_traces_per_order

    joined = pd.concat(frames, ignore_index=True, sort=False)
    joined_csv.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(joined_csv, index=False)


def run_visualizer(config, config_dir, joined_csv, output_dir):
    if not config.get("visualize", True):
        return

    visualizer = resolve_path(config.get("visualize_script", str(SCRIPT_DIR / "visualize_traces.py")), config_dir)
    plot_prefix = resolve_path(config.get("plot_prefix", str(output_dir / "traces_by_order")), config_dir)

    args = [
        sys.executable,
        str(visualizer),
        str(joined_csv),
        "--color-by",
        "order",
        "--out-prefix",
        str(plot_prefix),
    ]

    visualizer_options = config.get("visualizer", {})
    x_wall = visualizer_options.get("x_wall", config.get("x_wall"))
    r_vel = visualizer_options.get("r_vel", require(config, "radii")[0])
    r_acc = visualizer_options.get("r_acc", require(config, "radii")[1] if len(require(config, "radii")) >= 2 else None)

    if x_wall is not None:
        args.extend(["--x-wall", str(x_wall)])
    if r_vel is not None:
        args.extend(["--r-vel", str(r_vel)])
    if r_acc is not None:
        args.extend(["--r-acc", str(r_acc)])
    args.extend(["--radii", *(str(r) for r in require(config, "radii")[:require(config, "k")])])
    args.extend(["--x-init", str(optional_int(config, "x_init", 0))])
    args.extend(["--y-init", str(optional_int(config, "y_init", 0))])
    for rect in config.get("obstacles") or []:
        x_min, y_min, x_max, y_max = rect
        shrink_x = require(config, "vx_cmd_max") / 2.5
        shrink_y = require(config, "vy_cmd_max") / 2.5
        shrunk = [
            x_min + shrink_x,
            y_min + shrink_y,
            x_max - shrink_x,
            y_max - shrink_y,
        ]
        if shrunk[0] <= shrunk[2] and shrunk[1] <= shrunk[3]:
            args.extend(["--obstacle", *(str(value) for value in shrunk)])

    subprocess.run(args, check=True)


def main():
    ap = argparse.ArgumentParser(description="Run synthesis/simulation for orders 1..k and join traces")
    ap.add_argument("config", help="JSON config path")
    ap.add_argument("--resynthesize", action="store_true", help="Recompute strategies even when strategy files already exist")
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config_dir = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)

    k = require(config, "k")
    default_output_dir = SCRIPT_DIR / "experimental_results" / "order_sweep_outputs"
    output_dir = resolve_path(config.get("output_dir", str(default_output_dir)), config_dir)
    inputs_dir = output_dir / "inputs"
    strategies_dir = output_dir / "strategies"
    traces_dir = output_dir / "traces"
    logs_dir = output_dir / "logs"

    synth_exe = resolve_path(config.get("synth_exe", str(SCRIPT_DIR / "synth.o")), config_dir)
    sim_exe = resolve_path(config.get("sim_exe", str(SCRIPT_DIR / "sim.o")), config_dir)
    policy = policy_tokens(config)
    seed = require(config, "random_seed")
    seed_stride = int(config.get("seed_stride", 1))
    num_traces_per_order = require(config, "num_traces_per_order")
    resynthesize = args.resynthesize or bool(config.get("resynthesize", False))

    trace_paths = []
    for order in range(1, k + 1):
        strategy_path = strategies_dir / f"strategy_order_{order}.bin"
        synth_input_path = inputs_dir / f"synth_order_{order}.txt"
        sim_input_path = inputs_dir / f"sim_order_{order}.txt"
        trace_path = traces_dir / f"traces_order_{order}.csv"
        strategy_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        write_synthesis_input(synth_input_path, config, order, strategy_path)
        if strategy_path.exists() and not resynthesize:
            print(f"[order {order}] reusing existing strategy {strategy_path}", flush=True)
        else:
            print(f"[order {order}] synthesizing {strategy_path}", flush=True)
            run_with_input(synth_exe, synth_input_path, logs_dir / f"synth_order_{order}.log")

        order_seed = seed + (order - 1) * seed_stride
        write_simulation_input(sim_input_path, config, order, strategy_path, trace_path, order_seed, policy)
        print(f"[order {order}] simulating {trace_path}", flush=True)
        run_with_input(sim_exe, sim_input_path, logs_dir / f"sim_order_{order}.log")

        trace_paths.append((order, trace_path))

    joined_csv = resolve_path(config.get("joined_csv", str(output_dir / "traces_by_order.csv")), config_dir)
    join_traces(trace_paths, joined_csv, num_traces_per_order)
    print(f"wrote joined traces to: {joined_csv}", flush=True)

    run_visualizer(config, config_dir, joined_csv, output_dir)


if __name__ == "__main__":
    main()
