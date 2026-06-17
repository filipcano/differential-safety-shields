#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


SOLVERS = ["basic", "optimized", "optimized-iterative"]
TIME_REQUIRED_COLUMNS = ["benchmark_id", "solver_mode", "status", "elapsed_seconds", "k"]
MEMORY_REQUIRED_COLUMNS = ["benchmark_id", "solver_mode", "status", "peak_rss_mib", "k"]
name_dict = {"basic" : "Baseline", "optimized" : "Direct", "optimized-iterative":"Iterative" }
TABLE_COLUMN_SPACING_SCALE = "1.2"

def die(message):
    raise SystemExit(message)


def read_csv(path, required_columns):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            die(f"{path} is empty or has no CSV header")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            die(f"{path} is missing required columns: {', '.join(missing)}")
        rows = list(reader)
    return reader.fieldnames, rows


def indexed_rows(rows, source_name):
    index = {}
    for row in rows:
        key = (row["benchmark_id"], row["solver_mode"])
        if key in index:
            die(
                f"{source_name} contains duplicate rows for "
                f"benchmark_id={key[0]!r}, solver_mode={key[1]!r}"
            )
        index[key] = row
    return index


def benchmark_order(rows):
    seen = set()
    order = []
    for row in rows:
        benchmark_id = row["benchmark_id"]
        if benchmark_id in seen:
            continue
        seen.add(benchmark_id)
        order.append(benchmark_id)
    return order


def max_k_from_rows(rows):
    max_k = 0
    for row in rows:
        try:
            max_k = max(max_k, int(row["k"]))
        except ValueError:
            die(f"invalid k value for benchmark_id={row.get('benchmark_id', '')!r}: {row.get('k', '')!r}")
    return max_k


def pruning_orders(fieldnames, rows):
    pattern = re.compile(r"^iter_prune_fraction_d([0-9]+)$")
    orders = set()
    for fieldname in fieldnames:
        match = pattern.match(fieldname)
        if match:
            order = int(match.group(1))
            if order != 2:
                orders.add(order)
    if orders:
        return sorted(orders)
    max_k = max_k_from_rows(rows)
    return [order for order in range(1, max_k + 1) if order != 2]


def status_token(status):
    status = (status or "").strip()
    if status == "ok":
        return None
    if "timeout" in status:
        return "TO"
    if status == "error":
        return "ERR"
    return "--"


def format_float(value, precision, fixed=False):
    if value is None or value == "":
        return "--"
    try:
        number = float(value)
    except ValueError:
        return "--"
    if fixed:
        return f"{number:.{precision}f}"
    return f"{number:.{precision}g}"


def table_cell(row, metric_column, precision, fixed=False):
    if row is None:
        return "--"
    token = status_token(row.get("status", ""))
    if token is not None:
        return token
    return format_float(row.get(metric_column, ""), precision, fixed)


def metric_value(row, metric_column):
    if row is None or status_token(row.get("status", "")) is not None:
        return None
    value = row.get(metric_column, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def bold_latex(cell):
    return f"\\textbf{{{cell}}}"


def solver_metric_cells(benchmark_id, table_index, metric_column, precision, fixed=False):
    cells = []
    numeric_values = []
    for solver in SOLVERS:
        row = table_index.get((benchmark_id, solver))
        cells.append(latex_escape(table_cell(row, metric_column, precision, fixed)))
        numeric_values.append(metric_value(row, metric_column))

    valid_values = [value for value in numeric_values if value is not None]
    if not valid_values:
        return cells

    best = min(valid_values)
    return [
        bold_latex(cell) if value == best else cell
        for cell, value in zip(cells, numeric_values)
    ]


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


def latex_label(text):
    text = str(text)
    if text.startswith("$") and text.endswith("$"):
        return text
    return latex_escape(text)


def benchmark_param_label(benchmark_id, table_index):
    for solver in SOLVERS:
        row = table_index.get((benchmark_id, solver))
        if row is None:
            continue
        required = ["x_max", "y_max", "vx_cmd_max", "vy_cmd_max", "eta_max", "radii"]
        if not all(column in row and row[column] != "" for column in required):
            continue
        radii = ", ".join(row["radii"].replace(",", " ").split())
        return (
            f"$E^{{({row['x_max']}, {row['y_max']}), "
            f"({row['vx_cmd_max']}, {row['vy_cmd_max']}), "
            f"{row['eta_max']}}}_{{[{radii}]}}$"
        )
    return latex_label(benchmark_id)


def format_percent(value):
    if value == "-":
        return "-"
    if value is None or value == "":
        return r"0.0\%"
    try:
        fraction = float(value)
    except ValueError:
        return r"0.0\%"
    return f"{fraction * 100.0:.1f}" + r"\%"


def benchmark_k(benchmark_id, index):
    for solver in SOLVERS:
        row = index.get((benchmark_id, solver))
        if row is None:
            continue
        try:
            return int(row["k"])
        except ValueError:
            die(f"invalid k value for benchmark_id={benchmark_id!r}: {row.get('k', '')!r}")
    return 0


def pruning_row(benchmark_id, table_index, orders):
    row = table_index.get((benchmark_id, "optimized-iterative"))
    k = benchmark_k(benchmark_id, table_index)

    values = []
    for order in orders:
        if order > k:
            values.append("-")
            continue
        if row is None:
            values.append(r"0.0\%")
            continue
        column = f"iter_prune_fraction_d{order}"
        value = row.get(column, "")
        values.append(format_percent(value))
    return values


# def render_longtable(title, metric_column, precision, benchmark_ids, time_index, memory_index, orders, fixed=False):
#     data_index = time_index if metric_column == "elapsed_seconds" else memory_index
#     headers = ["Benchmark"] + SOLVERS + [f"$d={order}$" for order in orders]
#     column_spec = "l" + ("r" * (len(headers) - 1))

#     lines = []
#     lines.append(f"\\begin{{longtable}}{{{column_spec}}}")
#     lines.append(f"\\caption{{{latex_escape(title)}}}\\\\")
#     lines.append("\\toprule")
#     lines.append(" & ".join(latex_escape(header) if not header.startswith("$") else header for header in headers) + r" \\")
#     lines.append("\\midrule")
#     lines.append("\\endfirsthead")
#     lines.append("\\toprule")
#     lines.append(" & ".join(latex_escape(header) if not header.startswith("$") else header for header in headers) + r" \\")
#     lines.append("\\midrule")
#     lines.append("\\endhead")
#     lines.append("\\midrule")
#     lines.append(f"\\multicolumn{{{len(headers)}}}{{r}}{{Continued on next page}}\\\\")
#     lines.append("\\midrule")
#     lines.append("\\endfoot")
#     lines.append("\\bottomrule")
#     lines.append("\\endlastfoot")

#     for benchmark_id in benchmark_ids:
#         cells = [latex_escape(benchmark_id)]
#         for solver in SOLVERS:
#             cells.append(latex_escape(table_cell(data_index.get((benchmark_id, solver)), metric_column, precision, fixed)))
#         cells.extend(pruning_row(benchmark_id, time_index, memory_index, orders))
#         lines.append(" & ".join(cells) + r" \\")

#     lines.append("\\end{longtable}")
#     return "\n".join(lines)

def render_table(table_type, title, label, metric_column, precision, benchmark_ids, table_index, orders, fixed=False):
    display_orders = [order for order in orders if order > 1]
    headers = ["Benchmark"] + SOLVERS + [f"$d={order}$" for order in display_orders]
    # column_spec = "l" + ("r" * (len(headers) - 2)) ## -2 because we skip one
    column_spec = "l" + ("c" * len(SOLVERS))
    if display_orders:
        column_spec += "|" + ("r" * len(display_orders))
    lines = []
    lines.append(f"\\begin{{table}}")
    lines.append(f"\\centering")
    lines.append(f"\\setlength{{\\tabcolsep}}{{{TABLE_COLUMN_SPACING_SCALE}\\tabcolsep}}")
    lines.append(f"\\caption{{{latex_escape(title)}}}")
    lines.append(f"\\label{{{latex_escape(label)}}}")

    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")

    lines.append("\\toprule")    
    header_line = []
    header_line.append("\\multirow{2}{*}{Benchmark} ")
    
    if table_type == "time":
        header_line.append(f"\\multicolumn{{{len(SOLVERS)}}}{{c}}{{Synth. Time (s)}}")
    else:
        header_line.append(f"\\multicolumn{{{len(SOLVERS)}}}{{c}}{{Synth. Memory (MB)}}")

    if display_orders:
        header_line.append(f"\\multicolumn{{{len(display_orders)}}}{{c}}{{Pruned States (\\%)}}")
    header_line[-1] += "\\\\"
    lines.append(" & ".join(header_line))
    lines.append(f"\\cline{{{2}-{1 + len(SOLVERS) + len(display_orders)}}}")
    header_line = [" "]
    for solver in SOLVERS:
        header_line.append(f"{name_dict[solver]}")
    for order in display_orders:
        header_line.append(f"$d = {order}$")
    lines.append(" & ".join(header_line)+"\\\\")

    # lines.append(" & ".join(latex_escape(header) if not header.startswith("$") else header for header in headers) + r" \\")
    lines.append("\\midrule")
    for benchmark_id in benchmark_ids:
        cells = [benchmark_param_label(benchmark_id, table_index)]
        cells.extend(solver_metric_cells(benchmark_id, table_index, metric_column, precision, fixed))
        cells.extend(pruning_row(benchmark_id, table_index, display_orders))
        lines.append(" & ".join(cells) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)



# def write_tables(time_csv, memory_csv, output_tex):
def write_tables(table_type, table_csv, output_file, caption, label):
    if table_type == "time":
        fields, table_rows = read_csv(table_csv, TIME_REQUIRED_COLUMNS)        
    elif table_type == "memory":
        fields, table_rows = read_csv(table_csv, MEMORY_REQUIRED_COLUMNS)
    table_index = indexed_rows(table_rows, str(table_csv))
    benchmark_ids = benchmark_order(table_rows)
    orders = pruning_orders(list(fields), table_rows)

    precision = 3 if table_type == "time" else 1
    valname = "elapsed_seconds" if table_type == "time" else "peak_rss_mib"
    fixed = False if table_type =="time" else True
    table = render_table(table_type, caption, label,  valname, precision, benchmark_ids, table_index, orders, fixed=fixed)

    # tables = [
    #     "% Requires: \\usepackage{booktabs,longtable}",
    #     "",
    #     render_longtable(
    #         "Synthesis time by benchmark",
    #         "elapsed_seconds",
    #         3,
    #         benchmark_ids,
    #         time_index,
    #         memory_index,
    #         orders,
    #     ),
    #     "",
    #     render_longtable(
    #         "Peak synthesis memory by benchmark",
    #         "peak_rss_mib",
    #         1,
    #         benchmark_ids,
    #         time_index,
    #         memory_index,
    #         orders,
    #         fixed=True,
    #     ),
    #     "",
    # ]

    with open(output_file, "w") as fp:
        fp.write(table)


    # output_tex = Path(output_tex)
    # output_tex.parent.mkdir(parents=True, exist_ok=True)
    # output_tex.write_text("\n".join(tables), encoding="utf-8")


def main():

    ap = argparse.ArgumentParser(description="Export synthesis benchmark CSVs as LaTeX longtables")
    ap.add_argument("--table-type", help="Type of table, can be time or memory", type=str, 
                choices=["time", "memory"], default = "time")
    ap.add_argument("--input", help="CSV input file")
    ap.add_argument("--output", help="TeX output file")
    ap.add_argument("--caption", help="Table caption", default="Caption")
    ap.add_argument("--label", help="Table label", default="label")
    # ap.add_argument("time_csv", help="CSV produced by run_synthesis_benchmark.py for synthesis time")
    # ap.add_argument("memory_csv", help="CSV produced by run_synthesis_benchmark.py for synthesis memory")
    # ap.add_argument("output_tex", help="Output .tex file")
    args = ap.parse_args()

    # write_tables(args.time_csv, args.memory_csv, args.output_tex)
    write_tables(args.table_type, args.input, args.output, args.caption, args.label)


if __name__ == "__main__":
    main()
