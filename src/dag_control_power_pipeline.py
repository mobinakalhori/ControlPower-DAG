"""
Full pipeline: random network generation -> node power computation
-> degree-based node grouping per run -> two-level statistical averaging
-> power-vs-degree plots for three alpha/beta settings.

Grouping logic:
  1) In each run (each time a random network is generated), every node has
     an integer degree. Nodes are grouped by this integer degree within
     that same run.
  2) Within each run, for every degree group, the mean and standard
     deviation of node power are computed (first-level averaging - across
     nodes of equal degree).
  3) This is repeated N_REPEATS times. Since the network is random, the
     maximum degree observed can vary across runs.
  4) For each degree value, the per-run means (level 1) are averaged again
     across runs, with a corresponding standard deviation (second-level
     averaging - across runs). This final standard deviation is naturally
     larger than the within-run standard deviation, since it results from
     averaging twice.
  5) A single example run (the one with the lowest within-run standard
     deviation) is also saved separately, to show in the paper's
     supplementary material how much smaller the within-run variability
     is on its own.

Outputs (in this folder):
    - sample_adjacency_matrix.csv / .png
    - per_run_degree_summary_out.csv / _in.csv      (raw: per-run values)
    - aggregated_degree_summary_out.csv / _in.csv   (final: mean/std over 50 runs)
    - example_run_outdegree_supplementary.csv / indegree...
    - power_vs_outdegree_aggregated.png / power_vs_indegree_aggregated.png
    - power_vs_outdegree_examplerun.png / power_vs_indegree_examplerun.png
"""

import random
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Section 1: random network generation
# =========================================================

def generate_dag(num_nodes, avg_degree):
    """Build a random DAG with n nodes and a given average degree,
    returned as an adjacency list (dict)."""
    if num_nodes <= 1:
        return {i: [] for i in range(num_nodes)}

    p = avg_degree / (num_nodes - 1)
    if p > 1:
        p = 1.0

    adj_list = {i: [] for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < p:
                adj_list[i].append(j)
    return adj_list


def adjlist_to_matrix(adj_list, n):
    matrix = [[0] * n for _ in range(n)]
    for source, targets in adj_list.items():
        for target in targets:
            matrix[source][target] = 1
    return matrix


# =========================================================
# Section 2: node power computation
# =========================================================

def control_matrices(adj):
    n = len(adj)
    out_degree = [sum(adj[i]) for i in range(n)]

    parents = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                parents[j].append(i)

    matrices = {}
    q = deque()
    for i in range(n):
        if out_degree[i] == 0:
            q.append(i)

    removed = [False] * n

    while q:
        level_size = len(q)
        current_level = [q.popleft() for _ in range(level_size)]

        for v in current_level:
            if v not in matrices:
                matrices[v] = {"vertices": {v}, "count": {v: 1}, "dist": {v: 0}}

            for p in parents[v]:
                if removed[p]:
                    continue
                if p not in matrices:
                    matrices[p] = {"vertices": {p}, "count": {p: 1}, "dist": {p: 0}}

                for u in matrices[v]["vertices"]:
                    matrices[p]["vertices"].add(u)
                    matrices[p]["count"][u] = (
                        matrices[p]["count"].get(u, 0) + matrices[v]["count"][u]
                    )
                    d = matrices[v]["dist"][u] + 1
                    if u not in matrices[p]["dist"]:
                        matrices[p]["dist"][u] = d
                    else:
                        matrices[p]["dist"][u] = min(matrices[p]["dist"][u], d)

        for v in current_level:
            removed[v] = True
            for p in parents[v]:
                out_degree[p] -= 1
                if out_degree[p] == 0:
                    q.append(p)

    result = {}
    for v in range(n):
        verts = sorted(matrices[v]["vertices"])
        result[v] = [
            verts,
            [matrices[v]["count"][x] for x in verts],
            [matrices[v]["dist"][x] for x in verts],
        ]
    return result


def calculate_scores(result, alpha=1, beta=1):
    scores = {}
    for v in result:
        vertices = result[v][0]
        P = result[v][1]
        L = result[v][2]

        R = len(vertices)
        Pmax = max(P) or 1
        Lmax = max(L) or 1

        sumP = sum(p / Pmax for p in P)
        sumL = sum(l / Lmax for l in L)

        S = R + alpha * sumP - beta * sumL
        scores[v] = {"R": R, "Pmax": Pmax, "Lmax": Lmax, "Power": S}
    return scores


# =========================================================
# Section 3: sample adjacency matrix (for visual inspection)
# =========================================================

def save_sample_adjacency_matrix(n_nodes, avg_degree, csv_path, png_path):
    """Generate an independent random network, save its adjacency matrix
    with an out_degree column and an in_degree row, and plot an annotated
    heatmap (out-degree value shown next to each row)."""

    adj_list = generate_dag(n_nodes, avg_degree)
    matrix = np.array(adjlist_to_matrix(adj_list, n_nodes))

    out_degree = matrix.sum(axis=1)
    in_degree = matrix.sum(axis=0)

    labels = [f"Node_{i}" for i in range(n_nodes)]
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df["out_degree"] = out_degree
    in_degree_row = pd.Series(in_degree, index=labels)
    in_degree_row["out_degree"] = ""
    df.loc["in_degree"] = in_degree_row
    df.to_csv(csv_path, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(matrix, cmap="Blues", interpolation="none")
    ax.set_title(
        f"Sample Adjacency Matrix ({n_nodes} nodes, avg_degree={avg_degree})\n"
        "numbers on the right = out-degree (row sum)"
    )
    ax.set_xlabel("Target node (j)")
    ax.set_ylabel("Source node (i)")
    for i in range(n_nodes):
        ax.text(n_nodes + 1, i, str(int(out_degree[i])), va="center", ha="left", fontsize=7)
    ax.set_xlim(-0.5, n_nodes + 4)
    plt.colorbar(im, ax=ax, label="Edge (1) / No edge (0)", fraction=0.04)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Sample matrix saved: {csv_path}, {png_path}")


# =========================================================
# Section 4: single run with integer-degree grouping (first-level averaging)
# =========================================================

def compute_single_run(n_nodes, avg_degree, settings):
    """Generate a random network; compute the in-degree and out-degree of
    each node; group nodes by integer degree; compute each node's power
    under all three settings; and, for each degree group, return the
    mean/standard deviation of power within that group (averaged only
    across nodes of equal degree within this single run)."""

    adj_list = generate_dag(n_nodes, avg_degree)
    matrix = adjlist_to_matrix(adj_list, n_nodes)
    result = control_matrices(matrix)

    out_degree = [sum(row) for row in matrix]
    in_degree = [0] * n_nodes
    for i in range(n_nodes):
        for j in range(n_nodes):
            if matrix[i][j]:
                in_degree[j] += 1
    degrees = {"out": out_degree, "in": in_degree}

    power_by_setting = {}
    for name, (alpha, beta) in settings.items():
        scores = calculate_scores(result, alpha=alpha, beta=beta)
        power_by_setting[name] = [scores[v]["Power"] for v in range(n_nodes)]

    degree_groups = {dt: defaultdict(list) for dt in degrees}
    for dt, deg_list in degrees.items():
        for node_id, d in enumerate(deg_list):
            degree_groups[dt][d].append(node_id)

    within_run_stats = {dt: {name: {} for name in settings} for dt in degrees}
    for dt in degrees:
        for d, nodes in degree_groups[dt].items():
            for name in settings:
                vals = [power_by_setting[name][v] for v in nodes]
                within_run_stats[dt][name][d] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                    "node_ids": nodes,
                }

    return {"degree_groups": degree_groups, "within_run_stats": within_run_stats}


def repeated_degree_analysis(n_repeats, n_nodes, avg_degree, settings, verbose=True):
    records = []
    for r in range(n_repeats):
        records.append(compute_single_run(n_nodes, avg_degree, settings))
        if verbose and (r + 1) % max(1, n_repeats // 10) == 0:
            print(f"  ...run {r + 1}/{n_repeats} completed")
    return records


# =========================================================
# Section 5: second-level averaging (across runs) + raw per-run table
# =========================================================

def build_per_run_long_table(records, settings, degree_type):
    """Raw table: mean/standard deviation of each degree group, broken
    down by run (first-level averaging), together with the exact list of
    node IDs in each group."""
    rows = []
    for run_idx, rec in enumerate(records):
        stats = rec["within_run_stats"][degree_type]
        for name in settings:
            for d, s in sorted(stats[name].items()):
                rows.append({
                    "run": run_idx,
                    "setting": name,
                    "degree": d,
                    "n_nodes_in_group": s["n"],
                    "node_ids": ";".join(map(str, s["node_ids"])),
                    "within_run_mean_power": s["mean"],
                    "within_run_std_power": s["std"],
                })
    return pd.DataFrame(rows)


def aggregate_across_runs(records, settings, degree_type):
    """Second-level averaging: for each degree value, average the
    first-level means (one per run in which that degree was observed)
    and compute their standard deviation."""
    agg_means = {name: defaultdict(list) for name in settings}
    agg_group_sizes = {name: defaultdict(list) for name in settings}

    for rec in records:
        stats = rec["within_run_stats"][degree_type]
        for name in settings:
            for d, s in stats[name].items():
                agg_means[name][d].append(s["mean"])
                agg_group_sizes[name][d].append(s["n"])

    frames = {}
    for name in settings:
        rows = []
        for d in sorted(agg_means[name].keys()):
            means = agg_means[name][d]
            rows.append({
                "degree": d,
                "n_runs_with_this_degree": len(means),
                "avg_group_size": float(np.mean(agg_group_sizes[name][d])),
                "mean_power": float(np.mean(means)),
                "std_power": float(np.std(means)) if len(means) > 1 else 0.0,
            })
        frames[name] = pd.DataFrame(rows)
    return frames


def pick_example_run(records, settings, degree_type):
    """Select the run with the lowest average within-run standard
    deviation across its degree groups (over all settings) - suitable
    for the supplementary example."""
    best_idx, best_score = None, None
    for idx, rec in enumerate(records):
        stats = rec["within_run_stats"][degree_type]
        stds = [s["std"] for name in settings for s in stats[name].values() if s["n"] > 1]
        score = float(np.mean(stds)) if stds else float("inf")
        if best_score is None or score < best_score:
            best_score, best_idx = score, idx
    return best_idx


def example_run_frames(records, settings, degree_type, run_idx):
    stats = records[run_idx]["within_run_stats"][degree_type]
    frames, rows_all = {}, []
    for name in settings:
        rows = []
        for d, s in sorted(stats[name].items()):
            row = {
                "degree": d,
                "n_nodes_in_group": s["n"],
                "node_ids": ";".join(map(str, s["node_ids"])),
                "mean_power": s["mean"],
                "std_power": s["std"],
            }
            rows.append(row)
            rows_all.append({"setting": name, **row})
        frames[name] = pd.DataFrame(rows)
    return frames, pd.DataFrame(rows_all)


# =========================================================
# Section 6: power-vs-degree plots
# =========================================================

def subsample_by_degree(df, max_bars=14):
    """If the number of distinct degree values is large (e.g. dense
    networks), subsample at regular intervals to avoid overcrowding the
    plot, while preserving the overall trend."""
    if len(df) <= max_bars:
        return df
    step = int(np.ceil(len(df) / max_bars))
    return df.iloc[::step].reset_index(drop=True)


def plot_degree_power(frames_by_setting, settings, degree_label, out_path, title_suffix="", max_bars=14):
    fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 5))
    if len(settings) == 1:
        axes = [axes]

    for ax, name in zip(axes, settings):
        df = subsample_by_degree(frames_by_setting[name], max_bars)
        ax.bar(df["degree"], df["mean_power"], yerr=df["std_power"], capsize=3,
               color="steelblue", edgecolor="black", linewidth=0.5, width=0.7,
               error_kw={"elinewidth": 1.2})
        ax.set_title(f"Setting: {name}", fontweight="bold")
        ax.set_xlabel(degree_label, fontweight="bold")
        ax.set_ylabel("Mean Node Power", fontweight="bold")
        ax.set_xticks(df["degree"])
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"Node Power vs {degree_label} {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# =========================================================
# Section 7: full pipeline for a given network density (sparse/normal/dense)
# =========================================================

def run_density_pipeline(density_name, avg_degree, n_nodes, n_repeats, settings, density_title):
    """Run all previous steps (sample matrix, repeated analysis, raw
    table, final table, example run, individual plots) for a given
    avg_degree value, saving files with a density-name prefix
    (sparse/normal/dense). Returns the final aggregated-plots dictionary
    for use in the 9-panel combined figure."""

    print(f"\n=== {density_title} Network (N={n_nodes}, \u27e8k\u27e9={avg_degree}) ===")

    save_sample_adjacency_matrix(
        n_nodes, avg_degree,
        f"sample_adjacency_matrix_{density_name}.csv",
        f"sample_adjacency_matrix_{density_name}.png",
    )

    print(f"Starting {n_repeats} independent runs on random {n_nodes}-node networks...")
    records = repeated_degree_analysis(n_repeats, n_nodes, avg_degree, settings)

    aggregated = {}
    for degree_type, label in [("out", "Out-degree"), ("in", "In-degree")]:
        long_df = build_per_run_long_table(records, settings, degree_type)
        long_df.to_csv(f"per_run_degree_summary_{density_name}_{degree_type}.csv",
                        index=False, encoding="utf-8-sig")

        agg_frames = aggregate_across_runs(records, settings, degree_type)
        combined = []
        for name, df in agg_frames.items():
            df2 = df.copy()
            df2.insert(1, "setting", name)
            combined.append(df2)
        agg_df = pd.concat(combined, ignore_index=True)
        agg_df.to_csv(f"aggregated_degree_summary_{density_name}_{degree_type}.csv",
                       index=False, encoding="utf-8-sig")

        plot_degree_power(
            agg_frames, settings, label,
            f"power_vs_{degree_type}degree_{density_name}_aggregated.png",
            title_suffix=f"({density_title} Network, N={n_nodes}, mean & std across {n_repeats} repeats)",
        )

        example_idx = pick_example_run(records, settings, degree_type)
        example_frames, example_long = example_run_frames(records, settings, degree_type, example_idx)
        example_long.to_csv(f"example_run_{density_name}_{degree_type}degree_supplementary.csv",
                             index=False, encoding="utf-8-sig")
        plot_degree_power(
            example_frames, settings, label,
            f"power_vs_{degree_type}degree_{density_name}_examplerun.png",
            title_suffix=f"({density_title} Network, single example run #{example_idx})",
        )

        aggregated[degree_type] = agg_frames  # setting_name -> DataFrame(degree, mean_power, std_power, ...)

    return aggregated


# =========================================================
# Section 8: final combined figure - 3x3 matrix (density x alpha/beta setting)
# =========================================================

def plot_single_bar_panel(ax, df_out, df_in, color_out, color_in, max_bars=10):
    """One panel: grouped bar chart of out-degree/in-degree power with
    standard error. Index positions (not the actual degree values) are
    used on the x-axis so that bar width stays consistent across all rows
    (even when the degree range is much wider for dense networks); the
    axis labels still show the actual degree values."""
    df_out_s = subsample_by_degree(df_out, max_bars)
    df_in_s = subsample_by_degree(df_in, max_bars)

    all_degrees = sorted(set(df_out_s["degree"]).union(set(df_in_s["degree"])))
    pos = {d: i for i, d in enumerate(all_degrees)}

    x_out = [pos[d] for d in df_out_s["degree"]]
    x_in = [pos[d] for d in df_in_s["degree"]]

    width = 0.38
    bar_out = ax.bar([x - width / 2 for x in x_out], df_out_s["mean_power"], width=width,
                      yerr=df_out_s["std_power"], capsize=2.5, color=color_out,
                      edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 1})
    bar_in = ax.bar([x + width / 2 for x in x_in], df_in_s["mean_power"], width=width,
                     yerr=df_in_s["std_power"], capsize=2.5, color=color_in,
                     edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 1})

    ax.set_xticks(range(len(all_degrees)))
    ax.set_xticklabels([str(d) for d in all_degrees])
    ax.tick_params(axis="x", labelsize=11, rotation=0)
    ax.tick_params(axis="y", labelsize=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.grid(alpha=0.25, axis="y")
    return bar_out, bar_in


def save_individual_panels(aggregated_by_density, settings, setting_labels, densities,
                            color_out="#2E8B57", color_in="#7B2CBF", prefix="panel"):
    """Save each of the 9 plots individually (high resolution), as a
    fallback for manual arrangement in a slide/page if a custom grid
    layout is hard to produce directly from code."""
    setting_names = list(settings.keys())
    panel_letters = "ABCDEFGHI"
    idx = 0
    for density_name, n_nodes, avg_degree, density_title in densities:
        for setting_name in setting_names:
            fig, ax = plt.subplots(figsize=(6, 5))
            df_out = aggregated_by_density[density_name]["out"][setting_name]
            df_in = aggregated_by_density[density_name]["in"][setting_name]
            bar_out, bar_in = plot_single_bar_panel(ax, df_out, df_in, color_out, color_in)
            ax.set_xlabel("Degree", fontweight="bold")
            ax.set_ylabel("Node Power", fontweight="bold")
            ax.set_title(f"{panel_letters[idx]}. {density_title} DAG, N={n_nodes}; "
                         f"\u27e8k\u27e9={int(avg_degree)} \u2014 {setting_labels[setting_name]}",
                         fontweight="bold", fontsize=11.5)
            ax.legend([bar_out, bar_in], ["Out-degree", "In-degree"], fontsize=9, frameon=False)
            fig.tight_layout()
            fname = f"{prefix}_{panel_letters[idx]}_{density_name}_{setting_name}.png"
            fig.savefig(fname, dpi=250, bbox_inches="tight")
            plt.close(fig)
            idx += 1
    print(f"{idx} individual panel images saved (prefix='{prefix}_...').")


def plot_main_grid(aggregated_by_density, settings, setting_labels, densities, out_path, panel_prefix="panel"):
    """Main figure: a 3x3 grid of bar charts (density x alpha/beta
    setting), each cell containing two bars (green out-degree, purple
    in-degree) with standard error. Panel letters A..I are placed outside
    each subplot (top-left, with a small offset), spacing between
    subplots is generous, and a single shared legend covers the whole
    figure. The overall figure title is omitted (full description goes in
    the paper's caption)."""

    setting_names = list(settings.keys())
    nrows = len(densities)
    ncols = len(setting_names)

    color_out = "#2E8B57"
    color_in = "#7B2CBF"
    panel_letters = "ABCDEFGHI"

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 5.3 * nrows))
    bar_out = bar_in = None

    for i, (density_name, n_nodes, avg_degree, density_title) in enumerate(densities):
        for j, setting_name in enumerate(setting_names):
            ax = axes[i, j]
            df_out = aggregated_by_density[density_name]["out"][setting_name]
            df_in = aggregated_by_density[density_name]["in"][setting_name]
            bo, bi = plot_single_bar_panel(ax, df_out, df_in, color_out, color_in)
            if bar_out is None:
                bar_out, bar_in = bo, bi

            if i == nrows - 1:
                ax.set_xlabel("Degree", fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel("Node Power", fontsize=13, fontweight="bold")

    # generous spacing between subplots + room for outer labels and column headers
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.10, right=0.98, wspace=0.38, hspace=0.55)

    # panel letters A..I placed just outside the top-left corner of each subplot
    # (fixed offset for all panels)
    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            bbox = ax.get_position()
            fig.text(bbox.x0 - 0.018, bbox.y1 + 0.014, panel_letters[idx],
                      fontsize=15, fontweight="bold", va="bottom", ha="right")
            idx += 1

    # column header (alpha/beta setting) - offset further above the top row,
    # as a single label centered over the whole column (not just the top
    # subplot) to make clear it applies to the entire column
    top_row_y1 = max(axes[0, j].get_position().y1 for j in range(ncols))
    for j, setting_name in enumerate(setting_names):
        bbox_l = axes[0, j].get_position()
        x_center = bbox_l.x0 + bbox_l.width / 2
        fig.text(x_center, top_row_y1 + 0.045, setting_labels[setting_name],
                  fontsize=18, fontweight="bold", va="bottom", ha="center")

    # row label: DAG name + N and <k> on a single line
    for i, (density_name, n_nodes, avg_degree, density_title) in enumerate(densities):
        bbox = axes[i, 0].get_position()
        y_center = bbox.y0 + bbox.height / 2
        row_label = f"{density_title} DAG\nN={n_nodes}; \u27e8k\u27e9={int(avg_degree)}"
        fig.text(0.02, y_center, row_label, fontsize=12, fontweight="bold",
                  va="center", ha="center", rotation=90)

    fig.legend([bar_out, bar_in], ["Out-degree", "In-degree"],
               loc="upper center", bbox_to_anchor=(0.55, 0.975),
               ncol=2, fontsize=13, frameon=False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Main combined 3x3 bar-chart figure saved: {out_path}")

    # also save each panel individually (fallback for manual slide layout)
    save_individual_panels(aggregated_by_density, settings, setting_labels, densities,
                            color_out, color_in, prefix=panel_prefix)


# =========================================================
# Section 9: main entry point
# =========================================================

if __name__ == "__main__":
    N_NODES = 50
    N_REPEATS = 1000

    # column order exactly as specified: col1=(1,0.5), col2=(1,1), col3=(0.5,1)
    SETTINGS = {
        "alpha1_beta0.5": (1, 0.5),
        "alpha1_beta1": (1, 1),
        "alpha0.5_beta1": (0.5, 1),
    }
    SETTING_LABELS = {
        "alpha1_beta0.5": r"$\alpha=1,\ \beta=0.5$",
        "alpha1_beta1": r"$\alpha=1,\ \beta=1$",
        "alpha0.5_beta1": r"$\alpha=0.5,\ \beta=1$",
    }

    # row order exactly as specified: sparse -> intermediate-density -> dense
    DENSITIES = [
        ("sparse", 5.0, N_NODES, "Sparse"),
        ("normal", 10.0, N_NODES, "Intermediate-density"),
        ("dense", 20.0, N_NODES, "Dense"),
    ]

    aggregated_by_density = {}
    for density_name, avg_degree, n_nodes, density_title in DENSITIES:
        aggregated_by_density[density_name] = run_density_pipeline(
            density_name, avg_degree, n_nodes, N_REPEATS, SETTINGS, density_title
        )

    # plot_main_grid expects (density_name, n_nodes, avg_degree, density_title) per row
    DENSITIES_FOR_PLOT = [(d[0], d[2], d[1], d[3]) for d in DENSITIES]
    plot_main_grid(aggregated_by_density, SETTINGS, SETTING_LABELS, DENSITIES_FOR_PLOT, "main_figure_9panels.png")

    print("\nDone.")
