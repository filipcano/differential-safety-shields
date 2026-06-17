# RV26 Strategy Synthesis Tools

This folder contains a small pipeline:

1. synthesize a masked winning strategy,
2. simulate traces from that strategy,
3. visualize the generated traces.

## Build

Run:

```sh
sh compile.sh
```

This builds:

- `synth.o` from `synthesize_strategy_general.cpp`
- `sim.o` from `simulate_strategy_general.cpp`

Despite the `.o` suffix, these are executable programs.

## Synthesis

Program:

```sh
./synth.o < synth_input_general_example.txt
```

Input fields, line by line:

```text
k
solver_mode
x_max y_max
vx_cmd_max vy_cmd_max
eta_max
x_wall
R_1 R_2 ... R_k
output_strategy_path
[optional] obstacles count x_min y_min x_max y_max ...
```

`solver_mode` is one of:

- `basic`
- `optimized`
- `optimized-iterative`

Output:

- a binary strategy file, for example `strategy_general_example.bin`
- progress and cost information printed to stderr

The strategy file stores, for each winning encoded state, the mask of allowed safe actions.

For `basic`, synthesis uses an explicit-history state representation whenever
the packed `uint64_t` state id would overflow for the required history length.
These runs write strategy version 4, which the simulator also reads using
explicit-history lookups.

The wall is always active. To add forbidden rectangular obstacle regions, append
an optional tail after the strategy path:

```text
obstacles <count> <x_min> <y_min> <x_max> <y_max> ...
```

Rectangle coordinates are inclusive integer grid bounds.

## Simulation

Program:

```sh
./sim.o < sim_input_general_example.txt
```

Input fields, line by line:

```text
k
x_init y_init
x_max y_max
vx_cmd_max vy_cmd_max
eta_max
x_wall
R_1 R_2 ... R_k
strategy_path
num_traces
max_steps
output_traces_csv_path
random_seed
[optional policy]
[optional obstacles]
```

The problem parameters must match the strategy file produced by synthesis.
The full initial history is initialized to `x_init y_init`; use `0 0` for the
old default behavior.
If the strategy was synthesized with obstacles, the same obstacle tail must be
provided to simulation after any optional policy tokens.

Output:

- a CSV trace file, for example `traces_general_example.csv`

The CSV includes position, selected command, sampled noise, realized displacement, derivative values, and safety status:

```text
trace,step,x,y,cmd_vx,cmd_vy,noise_x,noise_y,real_dx,real_dy,D1_x,D1_y,...,safe,reason
```

If no optional policy is provided, the simulator samples uniformly from the synthesized action mask.

Supported optional policies:

```text
uniform
x_axis_max deterministic
x_axis_max softmax <temperature>
x_target_speed deterministic <target_speed>
x_target_speed softmax <target_speed> <temperature>
```

All policies are masked by the synthesized strategy: the simulator never chooses an action that synthesis marked unsafe.

## Visualization

Program:

```sh
python3 visualize_traces.py traces_general_example.csv
```

Useful options:

```sh
python3 visualize_traces.py traces_general_example.csv --x-wall 85 --r-vel 4 --r-acc 5
python3 visualize_traces.py traces_general_example.csv --out-prefix my_run
python3 visualize_traces.py traces_general_example.csv --color-by order
python3 visualize_traces.py traces_general_example.csv --color-by order --radii 8 4 6 9
python3 visualize_traces.py traces_general_example.csv --obstacle 34 0 46 4
python3 visualize_traces.py traces_general_example.csv --show
```

Output PDFs:

- `<prefix>_xy.pdf`
- `<prefix>_speed.pdf`
- `<prefix>_accel.pdf`
- `<prefix>_derivatives.pdf`
- `<prefix>_order_pressure.pdf`, for order-sweep CSVs with `--radii`
- `<prefix>_order_heatmaps.pdf`, for order-sweep CSVs with `--radii`

If `--out-prefix` is omitted, the CSV basename is used as the prefix.

`--color-by <column>` makes all traces with the same value in that CSV column use
the same color. This is useful for joined CSVs with an `order` column.
When the CSV is an order sweep and `--radii r1 ... rk` is provided, the
visualizer reconstructs every derivative order from the positions and emits
paper-oriented summaries of normalized derivative pressure and violations.

## Order Sweep

Program:

```sh
python3 run_order_sweep.py experimental_setups/order_sweep_config_example.json
```

By default, existing strategy files are reused. This avoids recomputing
synthesis, which is usually the expensive step. To force recomputation, either
run:

```sh
python3 run_order_sweep.py experimental_setups/order_sweep_config_example.json --resynthesize
```

or set `"resynthesize": true` in the JSON config.

The sweep script reads a JSON config with the same problem parameters as a
synthesis input. For a config with order `k`, it runs orders `1, 2, ..., k`.
At order `d`, it uses only the first `d` radii and writes:

- `inputs/synth_order_d.txt`
- `strategies/strategy_order_d.bin`
- `inputs/sim_order_d.txt`
- `traces/traces_order_d.csv`
- `logs/synth_order_d.log`
- `logs/sim_order_d.log`

It then joins all trace CSVs into one CSV with extra columns:

- `order`: the synthesis/simulation order that produced the row
- `trace_in_order`: the original trace id inside that order

The joined `trace` column is renumbered globally so traces from different
orders do not collide.

Important JSON fields:

```json
{
  "k": 4,
  "solver_mode": "optimized",
  "x_max": 100,
  "y_max": 100,
  "x_init": 0,
  "y_init": 0,
  "vx_cmd_max": 8,
  "vy_cmd_max": 8,
  "eta_max": 2,
  "x_wall": 85,
  "radii": [4, 5, 7, 8],
  "num_traces_per_order": 3,
  "max_steps": 200,
  "random_seed": 123456789,
  "policy": "x_target_speed softmax 10 10",
  "output_dir": "../experimental_results/order_sweep_outputs",
  "resynthesize": false
}
```

If `visualize` is true or omitted, the script calls `visualize_traces.py` on the
joined CSV with `--color-by order`.

Optional obstacles can be included as JSON rectangles. The sweep script appends
the same obstacle tail to the generated synthesis and simulation inputs and
draws the rectangles in the XY plot:

```json
{
  "obstacles": [[34, 0, 46, 4]]
}
```

Two tuned order-sweep configs are included:

```sh
python3 run_order_sweep.py experimental_setups/order_sweep_wall_config.json --resynthesize
python3 run_order_sweep.py experimental_setups/order_sweep_obstacle_config.json --resynthesize
```

They use `k = 4` with multiple traces per order. The wall case uses a larger
2D state space and softmax action selection to show a spread of wall-avoidance
trajectories. The obstacle case uses a larger middle obstacle and emphasizes
earlier, smoother upward avoidance for higher orders.

## Synthesis Benchmark Sweep

Program:

```sh
python3 run_synthesis_benchmark.py experimental_setups/synthesis_benchmark_config_example.json
```

This benchmarks `basic`, `optimized`, and `optimized-iterative` synthesis modes
over a JSON-defined set of synthesis inputs. Each run has a timeout, defaulting
to 60 seconds. The runner uses `tqdm` for its progress bar. Override the timeout
with:

```sh
python3 run_synthesis_benchmark.py experimental_setups/synthesis_benchmark_config_example.json --timeout-seconds 120
```

The script streams results immediately after each run or skip and shows progress
as synthesis calls complete:

- `experimental_results/synthesis_benchmark_outputs/synthesis_time.csv`
- `experimental_results/synthesis_benchmark_outputs/synthesis_memory.csv`

The memory CSV records peak RSS in MiB when synthesis completes. Timeout rows
have empty time/memory measurements and status `timeout`. The script only prints
extra lines for issues such as timeouts or solver errors.

Both CSVs also include `iter_prune_fraction_d1`, `iter_prune_fraction_d2`, ...
up to the maximum `k` in the benchmark set. These columns are nonzero only for
`optimized-iterative` rows when a previous-order filter rejects part of the
locally safe candidate region before fixed-point propagation.

To export the time and memory CSVs as LaTeX `longtable`s:

```sh
python3 export_synthesis_benchmark_table.py \
  experimental_results/synthesis_benchmark_outputs/synthesis_time.csv \
  experimental_results/synthesis_benchmark_outputs/synthesis_memory.csv \
  experimental_results/synthesis_benchmark_outputs/synthesis_tables.tex
```

The generated `.tex` file contains one table for elapsed synthesis time and one
for peak RSS, each with solver columns and pruning-fraction columns by order.

To avoid repeated timeouts, the script uses a partial order per solver. If a
solver times out on an easier benchmark, harder benchmarks are marked
`pruned_timeout` without running. A benchmark is treated as harder when it has
greater-or-equal `k`, `x_max`, `y_max`, command maxima, `eta_max`, `x_wall`,
and comparable radii.

Set `"x_wall_offset": 5` to derive `x_wall = x_max - 5` for every expanded
benchmark. This is useful when sweeping `x_max`.

The JSON can use a `base` plus Cartesian `sweep`:

```json
{
  "timeout_seconds": 60,
  "solvers": ["basic", "optimized", "optimized-iterative"],
  "x_wall_offset": 5,
  "base": {
    "k": 1,
    "x_max": 10,
    "y_max": 10,
    "vx_cmd_max": 3,
    "vy_cmd_max": 3,
    "eta_max": 1,
    "radii": [4, 5, 6]
  },
  "sweep": {
    "k": [1, 2, 3],
    "x_max": [10, 15],
    "y_max": [10, 15],
    "vx_cmd_max": [3, 4],
    "vy_cmd_max": [3, 4],
    "eta_max": [1, 2],
    "radii": [[4, 5, 6], [5, 6, 7]]
  }
}
```

Alternatively, provide an explicit `benchmarks` list with the same synthesis
parameters.

For `optimized-iterative`, the synthesizer logs per-stage candidate/winning
counts, fixed-point losses, previous-filter rejections, and stage cost. Cases
where iterative tends to help have a lower-order stage, often `D2`, that removes
many states before the final higher-order solve. See
`experimental_setups/synthesis_benchmark_iterative_config_example.json` for representative pruning,
weak-pruning, and no-pruning benchmark cases. The `k=2` cases are large
tight-`D2` baselines; iterative is not expected to help much there because the
first useful pruning order is also the final order.

## Example Files

- `synth_input_general_example.txt`: example synthesis input.
- `sim_input_general_example.txt`: example simulation input.
- `experimental_setups/order_sweep_config_example.json`: example order-sweep config.
- `experimental_setups/synthesis_benchmark_config_example.json`: example synthesis benchmark config.
- `experimental_setups/synthesis_benchmark_iterative_config_example.json`: representative benchmark cases for iterative synthesis.
- `synth_input_fields.txt`: synthesis input field names.
- `sim_input_fields.txt`: simulation input field names.
- `run_order_sweep.py`: runs synthesis, simulation, CSV joining, and visualization for orders `1..k`.
- `run_synthesis_benchmark.py`: benchmarks synthesis time and memory across solver modes and settings.
- `strategy_general_example.bin`: generated strategy file.
- `traces_general_example.csv`: generated simulation traces.
- `traces_general_example_*.pdf`: generated trace plots.
