#!/usr/bin/env python3
import argparse
import math
import os
import re
import tempfile

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "rv26_plot_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)

import numpy as np
import pandas as pd

plt = None
ORDER_COLORS = [
    "#3B6EA8",
    "#C46A2D",
    "#2F8F62",
    "#8E5EA2",
    "#6F6F6F",
    "#B44E6B",
]


def finish(path, show):
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=180, bbox_inches="tight")
        print(f"wrote {path}")
    if show:
        plt.show()
    plt.close()


def require_columns(df, csv_path, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {csv_path}: {missing}")


def derivative_orders(df):
    orders = set()
    pattern = re.compile(r"^D([0-9]+)_([xy])$")
    for col in df.columns:
        match = pattern.match(col)
        if match:
            orders.add(int(match.group(1)))
    return sorted(d for d in orders if f"D{d}_x" in df.columns and f"D{d}_y" in df.columns)


def find_vector_columns(df, order, fallbacks):
    dx = f"D{order}_x"
    dy = f"D{order}_y"
    if dx in df.columns and dy in df.columns:
        return dx, dy, f"D{order}"
    for label, x_col, y_col in fallbacks:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col, label
    return None


def add_norm_column(df, x_col, y_col, out_col):
    df[out_col] = np.hypot(df[x_col], df[y_col])


def trace_identity_column(df):
    return "trace_in_order" if "trace_in_order" in df.columns else "trace"


def order_color(order, orders):
    try:
        idx = list(orders).index(order)
    except ValueError:
        idx = 0
    return ORDER_COLORS[idx % len(ORDER_COLORS)]


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


def norm_ratio(norm, radius):
    if radius == 0:
        return 0.0 if norm == 0 else np.inf
    return norm / radius


def reconstructed_derivative_frame(df, radii, x_init, y_init):
    require_columns(df, "trace CSV", ["order", "step", "x", "y"])
    max_order = len(radii)
    trace_col = trace_identity_column(df)
    records = []

    for shield_order, by_order in df.groupby("order", sort=True):
        for trace_id, trace in by_order.groupby(trace_col, sort=True):
            history = [(x_init, y_init) for _ in range(max_order)]
            trace = trace.sort_values("step")
            for row in trace.itertuples(index=False):
                row_dict = row._asdict()
                if str(row_dict.get("reason", "")) == "no_allowed_action":
                    continue
                next_pos = (int(row_dict["x"]), int(row_dict["y"]))
                step = int(row_dict["step"])
                for constraint_order in range(1, max_order + 1):
                    dx, dy = derivative_at_next(next_pos, history, constraint_order)
                    norm = float(np.hypot(dx, dy))
                    ratio = norm_ratio(norm, float(radii[constraint_order - 1]))
                    records.append({
                        "shield_order": int(shield_order),
                        "trace": int(trace_id),
                        "step": step,
                        "constraint_order": constraint_order,
                        "norm": norm,
                        "ratio": ratio,
                        "violation": ratio > 1.0,
                    })
                history = [next_pos] + history[:-1]

    if not records:
        return pd.DataFrame(columns=[
            "shield_order",
            "trace",
            "step",
            "constraint_order",
            "norm",
            "ratio",
            "violation",
        ])
    return pd.DataFrame.from_records(records)


def paper_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_order_pressure(deriv_df, radii, prefix, show):
    if deriv_df.empty:
        return

    axis_label_size = plt.rcParams["axes.labelsize"] * 1.2
    legend_font_size = plt.rcParams["legend.fontsize"] * 1.2
    legend_title_size = legend_font_size
    derivative_names = {
        1: "Velocity",
        2: "Acceleration",
        3: "Jerk",
        4: "Snap",
    }
    orders = sorted(deriv_df["shield_order"].unique())
    constraint_orders = list(range(1, len(radii) + 1))
    ncols = min(2, len(constraint_orders))
    nrows = int(np.ceil(len(constraint_orders) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(9.8, 2.35 * nrows),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    mean_cache = {}
    panel_ymax = {}
    for constraint_order in constraint_orders:
        mean_max = 1.0
        subset = deriv_df[deriv_df["constraint_order"] == constraint_order]
        for shield_order in orders:
            g = subset[subset["shield_order"] == shield_order]
            if g.empty:
                continue
            grouped = g.groupby("step")["ratio"]
            mean = grouped.mean()
            mean_cache[(shield_order, constraint_order)] = mean
            finite = mean.replace([np.inf, -np.inf], np.nan).dropna()
            if not finite.empty:
                mean_max = max(mean_max, float(finite.max()))
        panel_ymax[constraint_order] = max(1.25, min(6.0, mean_max * 1.12))

    for ax, constraint_order in zip(axes_flat, constraint_orders):
        y_max = panel_ymax[constraint_order]
        ax.axhspan(1.0, y_max, color="#F6D7BE", alpha=0.35, linewidth=0)
        ax.axhline(1.0, color="#8A2D20", linestyle="--", linewidth=0.9)
        subset = deriv_df[deriv_df["constraint_order"] == constraint_order]
        for shield_order in orders:
            g = subset[subset["shield_order"] == shield_order]
            if g.empty:
                continue
            mean = mean_cache[(shield_order, constraint_order)]
            color = order_color(shield_order, orders)
            linewidth = 1.65 if shield_order == constraint_order else 1.15
            linestyle = "-" if shield_order >= constraint_order else (0, (3, 2))
            ax.plot(
                mean.index,
                mean.values,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                label=f"$k={shield_order}$",
            )
        name = derivative_names.get(constraint_order, f"Derivative")
        ax.set_title(f"{name} ($d={constraint_order}$)")
        ax.set_ylim(0, y_max)
        ax.set_ylabel(r"$\rho_d$", fontsize=axis_label_size)
        ax.grid(True, axis="y", linewidth=0.35, color="#D8D8D8")
        ax.grid(True, axis="x", linewidth=0.25, color="#ECECEC")

    for ax in axes_flat[len(constraint_orders):]:
        ax.axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("step", fontsize=axis_label_size)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            ncol=1,
            frameon=False,
            title="Shield",
            fontsize=legend_font_size,
            title_fontsize=legend_title_size,
            bbox_to_anchor=(0.802, 0.5),
        )
    fig.subplots_adjust(right=0.785 if handles else 0.96, bottom=0.10, hspace=0.28, wspace=0.24)
    path = f"{prefix}_order_pressure.pdf" if prefix else None
    if path:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"wrote {path}")
    if show:
        plt.show()
    plt.close(fig)


def matrix_from_series(series, orders, constraint_orders, default=0.0):
    matrix = np.zeros((len(orders), len(constraint_orders)), dtype=float)
    matrix[:] = default
    for i, shield_order in enumerate(orders):
        for j, constraint_order in enumerate(constraint_orders):
            key = (shield_order, constraint_order)
            if key in series.index:
                matrix[i, j] = float(series.loc[key])
    return matrix


def annotate_heatmap(ax, matrix, fmt):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "0" if abs(value) < 1e-12 else fmt.format(value)
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color="#1F1F1F")


def plot_order_heatmaps(deriv_df, radii, prefix, show):
    if deriv_df.empty:
        return

    orders = sorted(deriv_df["shield_order"].unique())
    constraint_orders = list(range(1, len(radii) + 1))
    per_trace = deriv_df.groupby(["shield_order", "trace", "constraint_order"])
    peak_series = per_trace["ratio"].max().groupby(["shield_order", "constraint_order"]).mean()
    violation_series = per_trace["violation"].sum().groupby(["shield_order", "constraint_order"]).mean()
    peak = matrix_from_series(peak_series, orders, constraint_orders)
    violations = matrix_from_series(violation_series, orders, constraint_orders)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.25))
    peak_max = max(1.2, float(np.nanmax(peak)))
    try:
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=peak_max)
    except Exception:
        norm = None

    im0 = axes[0].imshow(peak, cmap="coolwarm", norm=norm, aspect="auto")
    axes[0].set_title("Mean peak pressure")
    annotate_heatmap(axes[0], peak, "{:.2f}")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label(r"$\max_t \|D^d\|/r_d$")

    vmax_viol = max(1.0, float(np.nanmax(violations)))
    im1 = axes[1].imshow(violations, cmap="OrRd", vmin=0.0, vmax=vmax_viol, aspect="auto")
    axes[1].set_title("Violations per trace")
    annotate_heatmap(axes[1], violations, "{:.1f}")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("count")

    for ax in axes:
        ax.set_xticks(range(len(constraint_orders)), [f"$d={d}$" for d in constraint_orders])
        ax.set_yticks(range(len(orders)), [f"$k={k}$" for k in orders])
        ax.set_xlabel("constraint order")
        ax.set_ylabel("shield order")
        ax.set_xticks(np.arange(-0.5, len(constraint_orders), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(orders), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle("Order sweep derivative summary", fontsize=11)
    finish(f"{prefix}_order_heatmaps.pdf" if prefix else None, show)


def make_trace_style_helpers(df, color_by):
    if color_by is None:
        def trace_color(_tid):
            return None

        def trace_label(tid, _g, _seen):
            return f"trace {tid}"

        return trace_color, trace_label, 0

    if color_by not in df.columns:
        raise SystemExit(f"Missing color grouping column in CSV: {color_by}")

    trace_to_group = {}
    for tid, g in df.groupby("trace", sort=False):
        values = g[color_by].dropna().unique()
        if len(values) != 1:
            raise SystemExit(f"Trace {tid} has multiple values for color grouping column {color_by}")
        trace_to_group[tid] = values[0]

    groups = sorted(set(trace_to_group.values()), key=lambda x: (str(type(x)), x))
    cmap = plt.get_cmap("tab20" if len(groups) > 10 else "tab10")
    colors = {group: cmap(i % cmap.N) for i, group in enumerate(groups)}

    def trace_color(tid):
        return colors[trace_to_group[tid]]

    def trace_label(tid, _g, seen):
        group = trace_to_group[tid]
        if group in seen:
            return None
        seen.add(group)
        return f"{color_by} {group}"

    return trace_color, trace_label, len(groups)


def main():
    global plt

    ap = argparse.ArgumentParser(description="Visualize game traces produced by simulate_strategy_general.cpp")
    ap.add_argument("csv", nargs="?", default="traces_general_example.csv")
    ap.add_argument("--out-prefix", default=None, help="Prefix for saved PDFs. Default: CSV basename")
    ap.add_argument("--x-wall", type=float, default=None, help="Optional x wall to draw")
    ap.add_argument("--r-vel", type=float, default=None, help="Optional D1/velocity bound to draw")
    ap.add_argument("--r-acc", type=float, default=None, help="Optional D2/acceleration bound to draw")
    ap.add_argument("--radii", nargs="+", type=float, default=None, help="Derivative radii r1 ... rk for order-sweep summary figures")
    ap.add_argument("--x-init", type=int, default=0, help="Initial x position used to reconstruct derivatives")
    ap.add_argument("--y-init", type=int, default=0, help="Initial y position used to reconstruct derivatives")
    ap.add_argument("--color-by", default=None, help="Column whose value controls trace color, e.g. order")
    ap.add_argument("--no-order-summary", action="store_true", help="Skip order-sweep derivative summary figures")
    ap.add_argument(
        "--obstacle",
        nargs=4,
        type=float,
        action="append",
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Obstacle rectangle to draw on the XY plot; can be repeated",
    )
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    args = ap.parse_args()

    import matplotlib
    if not args.show and "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt_module
    plt = plt_module
    paper_style()

    df = pd.read_csv(args.csv)
    require_columns(df, args.csv, ["trace", "step", "x", "y"])

    df = df.sort_values(["trace", "step"])

    d1_cols = find_vector_columns(df, 1, [
        ("real displacement", "real_dx", "real_dy"),
        ("velocity", "real_vx", "real_vy"),
    ])
    d2_cols = find_vector_columns(df, 2, [
        ("acceleration", "acc_x", "acc_y"),
    ])

    if d1_cols is None:
        raise SystemExit(
            f"Missing D1 vector columns in {args.csv}: expected D1_x/D1_y, "
            "real_dx/real_dy, or real_vx/real_vy"
        )

    add_norm_column(df, d1_cols[0], d1_cols[1], "d1_norm")
    if d2_cols is not None:
        add_norm_column(df, d2_cols[0], d2_cols[1], "d2_norm")

    all_orders = derivative_orders(df)
    for order in all_orders:
        add_norm_column(df, f"D{order}_x", f"D{order}_y", f"D{order}_norm")

    prefix = args.out_prefix
    if prefix is None and not args.show:
        prefix = os.path.splitext(args.csv)[0]

    trace_color, trace_label, _color_group_count = make_trace_style_helpers(df, args.color_by)
    trace_alpha = 0.5

    # 1. XY trajectories.
    xy_axis_font_size = plt.rcParams["axes.labelsize"] * 1.5
    xy_tick_font_size = plt.rcParams["xtick.labelsize"] * 1.5
    xy_legend_font_size = plt.rcParams["legend.fontsize"] * 1.5
    plt.figure(figsize=(8.05, 5))
    seen_labels = set()
    for tid, g in df.groupby("trace"):
        color = trace_color(tid)
        label = trace_label(tid, g, seen_labels)
        plt.plot(g["x"], g["y"], marker="o", markersize=2, linewidth=1, color=color, label=label, alpha=trace_alpha)
        plt.scatter(g["x"].iloc[0], g["y"].iloc[0], marker="s", color=color, alpha=trace_alpha)
        plt.scatter(g["x"].iloc[-1], g["y"].iloc[-1], marker="x", color=color, alpha=trace_alpha)
    if args.x_wall is not None:
        plt.axvline(args.x_wall, linestyle="--", linewidth=1, label="x wall")
    if args.obstacle:
        from matplotlib.patches import Rectangle
        obstacle_label_added = False
        for x_min, y_min, x_max, y_max in args.obstacle:
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            label = "obstacle" if not obstacle_label_added else None
            obstacle_label_added = True
            plt.gca().add_patch(Rectangle(
                (x_min - 0.5, y_min - 0.5),
                width,
                height,
                facecolor="black",
                edgecolor="black",
                alpha=0.18,
                linewidth=1,
                label=label,
            ))
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("x", fontsize=xy_axis_font_size)
    plt.ylabel("y", fontsize=xy_axis_font_size)
    plt.tick_params(axis="both", labelsize=xy_tick_font_size)
    plt.grid(True, linewidth=0.3)
    if args.color_by is not None or df["trace"].nunique() <= 12:
        plt.legend(fontsize=xy_legend_font_size)
    finish(f"{prefix}_xy.pdf" if prefix else None, args.show)

    # 2. Speed norm over time.
    plt.figure(figsize=(7, 4))
    seen_labels = set()
    for tid, g in df.groupby("trace"):
        color = trace_color(tid)
        label = trace_label(tid, g, seen_labels)
        plt.plot(g["step"], g["d1_norm"], linewidth=1, color=color, label=label, alpha=trace_alpha)
    if args.r_vel is not None:
        plt.axhline(args.r_vel, linestyle="--", linewidth=1, label="D1 bound")
    plt.xlabel("step")
    plt.ylabel(f"||{d1_cols[2]}||")
    plt.title(f"{d1_cols[2]} norm")
    plt.grid(True, linewidth=0.3)
    if args.color_by is not None or df["trace"].nunique() <= 12:
        plt.legend()
    finish(f"{prefix}_speed.pdf" if prefix else None, args.show)

    # 3. Acceleration norm over time.
    if d2_cols is not None:
        plt.figure(figsize=(7, 4))
        seen_labels = set()
        for tid, g in df.groupby("trace"):
            color = trace_color(tid)
            label = trace_label(tid, g, seen_labels)
            plt.plot(g["step"], g["d2_norm"], linewidth=1, color=color, label=label, alpha=trace_alpha)
        if args.r_acc is not None:
            plt.axhline(args.r_acc, linestyle="--", linewidth=1, label="D2 bound")
        plt.xlabel("step")
        plt.ylabel(f"||{d2_cols[2]}||")
        plt.title(f"{d2_cols[2]} norm")
        plt.grid(True, linewidth=0.3)
        if args.color_by is not None or df["trace"].nunique() <= 12:
            plt.legend()
        finish(f"{prefix}_accel.pdf" if prefix else None, args.show)

    # 4. All derivative norms, when the general simulator emitted D*_x/D*_y columns.
    if all_orders:
        if args.radii is not None and "order" in df.columns and not args.no_order_summary:
            deriv_df = reconstructed_derivative_frame(df, args.radii, args.x_init, args.y_init)
            plot_order_pressure(deriv_df, args.radii, prefix, args.show)
            plot_order_heatmaps(deriv_df, args.radii, prefix, args.show)
        else:
            plt.figure(figsize=(8, 5))
            for order in all_orders:
                by_step = df.groupby("step")[f"D{order}_norm"].max()
                plt.plot(by_step.index, by_step.values, linewidth=1.5, label=f"D{order}")
            if args.r_vel is not None and 1 in all_orders:
                plt.axhline(args.r_vel, linestyle="--", linewidth=1, label="D1 bound")
            if args.r_acc is not None and 2 in all_orders:
                plt.axhline(args.r_acc, linestyle=":", linewidth=1, label="D2 bound")
            plt.xlabel("step")
            plt.ylabel("max norm across traces")
            plt.title("Derivative norms")
            plt.grid(True, linewidth=0.3)
            plt.legend()
            finish(f"{prefix}_derivatives.pdf" if prefix else None, args.show)


if __name__ == "__main__":
    main()
