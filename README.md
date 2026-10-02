# ControlPower-DAG

Reference implementation of **ControlPower**, a bottom-up dynamic-programming algorithm that scores how much control each vertex of a directed acyclic graph (DAG) has over the rest of the network. The code accompanies the manuscript:

> Hashemi Dulabi B.S.\*, Kalhori M.\*, Afzali Nejad S.\*, Ebrahimi A.† **ControlPower: Bottom-Up Dynamic Programming Algorithm for Quantifying Vertex Control Strength in DAGs.** Department of Computer Science, Faculty of Mathematical Sciences, Alzahra University, Tehran, Iran.
> (\*equal contribution, †corresponding author)

---

## Overview

Most centrality measures capture a single facet of influence: local connectivity, shortest-path position, or membership in a minimum driver set. In a signaling pathway, a protein matters because it can reach many downstream targets, through more than one route, and ideally through short ones. ControlPower combines these three aspects (reach, redundancy and efficiency) in one score.

For a vertex *v* in a DAG:

- *R(v)* is the set of vertices reachable from *v*, including *v* itself;
- *P(v, u)* is the number of distinct directed paths from *v* to *u*, with *P(v, v) = 1*;
- *L(v, u)* is the length (in edges) of the shortest path from *v* to *u*, with *L(v, v) = 0*.

The control-power score is

$$
S(v) = |R(v)| + \alpha \sum_{u \in R(v)} \frac{P(v,u)}{P_{\max}(v)} - \beta \sum_{u \in R(v)} \frac{L(v,u)}{L_{\max}(v)}
$$

where *P*<sub>max</sub>(*v*) and *L*<sub>max</sub>(*v*) are the largest path count and the largest shortest-path length over *u* ∈ *R(v)* (each is replaced by 1 when it is 0, as for a sink). The non-negative parameters α and β set the weight given to path redundancy and to path length, respectively. A sink therefore scores 1 + α.

### How it is computed

Because the graph has no directed cycles, all tables can be built in a single sweep from the sinks upward:

1. Every sink gets the trivial table *R(v) = {v}*, *P(v, v) = 1*, *L(v, v) = 0*.
2. Sinks are removed layer by layer (a Kahn-style peeling of the reversed graph). When a vertex is reached, the tables of its successors are merged into its own: the reachable sets are united, path counts are added, and shortest-path lengths are updated with the minimum of the current value and *L(successor, u) + 1*.
3. Once every vertex has a table, *S(v)* is evaluated.

Each vertex is solved exactly once and the results of its successors are reused, so the output is exact.

| | |
|---|---|
| Time | O(*nm*): O(*n*²) on sparse graphs, O(*n*³) in the dense worst case |
| Space | O(*n*²) |
| Output | exact scores, no approximation |

The implementation is plain Python and works on a dense adjacency matrix, so it is intended for pathway-sized networks (tens to a few hundred vertices) rather than very large graphs.

---

## Repository layout

```
.
├── src/
│   ├── dag_reduction.py               # directed graph -> DAG (isolated nodes, SCC contraction)
│   ├── simulated_network.py           # random DAG generator, CSV/Excel export, visual check
│   └── dag_control_power_pipeline.py  # control-power computation + 1000-replicate experiment (Figure 1)
├── README.md
└── LICENSE
```

---

## Installation

Python 3.9 or newer.

```bash
git clone https://github.com/mobinakalhori/ControlPower-DAG.git
cd ControlPower-DAG
pip install numpy pandas matplotlib networkx openpyxl
```

---

## Usage

### 1. Convert a pathway graph to a DAG

Pathway maps from resources such as KEGG usually contain feedback loops, so they must be made acyclic before scoring. `dag_reduction.py` does this in two steps. It first removes isolated vertices, then contracts every strongly connected component (SCC) into a single super-vertex. Edges between different components are kept, so reachability between them is preserved. Isolated vertices created by the contraction are removed as well.

```bash
python src/dag_reduction.py \
    --input pathway_matrix.tsv \
    --input-format matrix \
    --output-list pathway_dag.txt \
    --output-matrix pathway_dag.tsv \
    --verbose
```

Supported inputs:

- `matrix`: a square, tab-separated adjacency matrix with node names in the first row and first column;
- `edgelist`: two or three columns (`source  target  [weight]`); add `--has-header` if the file starts with a header row.

Other flags: `--keep-isolated` keeps isolated vertices, `--sep` changes the field separator, `--name-sep` changes the separator used to name merged vertices (default `;`).

The script reports the size of the resulting DAG and lists which original vertices were merged into each super-vertex. At least one of `--output-list` or `--output-matrix` is required.

### 2. Compute control power

The scoring functions `control_matrices` and `calculate_scores` are in `src/dag_control_power_pipeline.py`. They take a 0/1 adjacency matrix (`adj[i][j] = 1` means an edge *i* → *j*) that must be acyclic.

A small example, with edges 0→1, 0→2, 1→3, 2→3:

```python
from dag_control_power_pipeline import control_matrices, calculate_scores

adj = [
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
]

tables = control_matrices(adj)
scores = calculate_scores(tables, alpha=1.0, beta=1.0)

for v, s in sorted(scores.items(), key=lambda kv: -kv[1]["Power"]):
    print(v, s["R"], round(s["Power"], 2))
```

With α = β = 1 this prints vertex 0 first (|R| = 4, S = 4.50), then vertices 1 and 2 (|R| = 2, S = 3.00 each) and finally the sink, vertex 3 (|R| = 1, S = 2.00). For each vertex, `calculate_scores` returns the reachable-set size (`R`), the two normalizers (`Pmax`, `Lmax`) and the score (`Power`).

To score a network produced in step 1, read the matrix file and map the vertex indices back to names (run this from `src/`, or add `src/` to `PYTHONPATH`):

```python
import pandas as pd
from dag_control_power_pipeline import control_matrices, calculate_scores

df = pd.read_csv("pathway_dag.tsv", sep="\t", index_col=0)
names = list(df.index)

scores = calculate_scores(control_matrices(df.values.tolist()), alpha=1.0, beta=1.0)
ranking = sorted(((names[v], s["Power"]) for v, s in scores.items()), key=lambda x: -x[1])

for name, power in ranking[:5]:
    print(f"{name}\t{power:.2f}")
```

### 3. Generate a random DAG

```bash
python src/simulated_network.py
```

The script asks for the number of nodes and the average degree ⟨k⟩. It saves the edge list (CSV) and the 0/1 adjacency matrix (Excel) to the Desktop folder, checks acyclicity with NetworkX and draws the graph. Close the plot window to let it continue with a short timing benchmark (100 networks of 50 nodes).

Acyclicity is guaranteed by construction. Vertices are ordered 0, …, *N*−1 and an edge *i* → *j* is allowed only if *i* < *j*; each allowed edge is drawn independently with probability *p* = ⟨k⟩ / (*N* − 1). Since each edge adds one to the out-degree of one vertex and one to the in-degree of another, ⟨k⟩ is the mean total degree (in + out), and the mean out-degree is about ⟨k⟩ / 2.

### 4. Reproduce the simulation experiment (Section 3.1)

```bash
python src/dag_control_power_pipeline.py
```

This runs the experiment described in the paper: *N* = 50 vertices, 1000 random DAGs for each of three densities (⟨k⟩ = 5, 10, 20), and three weightings (α, β) ∈ {(1, 0.5), (1, 1), (0.5, 1)}. It usually finishes within a few minutes on a laptop. The number of replicates (`N_REPEATS`) and the network size (`N_NODES`) are set at the bottom of the script, in the `__main__` block, if you want a quicker pass.

Averaging is done in two stages:

1. **Within a run.** Nodes are grouped by their integer degree, and the mean and standard deviation of power are computed inside each group.
2. **Across runs.** For each degree value, the per-run group means are averaged over all runs in which that degree occurred.

The error bars in the main figure come from the second stage, so they are larger than the within-run spread. For the supplementary material the script also saves one example run, chosen as the run with the lowest mean within-run standard deviation. This run is meant to show how small the within-run variability can be. It is the best case, not a typical one.

No random seed is fixed, so the exact numbers change slightly from one execution to the next. The qualitative patterns described below do not.

Outputs are written to the working directory:

| File | Content |
|---|---|
| `main_figure_9panels.png` | 3 × 3 grid (density × parameter setting), Figure 1 |
| `panel_*.png` | the nine panels as separate high-resolution images |
| `aggregated_degree_summary_<density>_<in/out>.csv` | final mean and standard deviation over runs |
| `per_run_degree_summary_<density>_<in/out>.csv` | raw per-run group statistics, with node IDs |
| `example_run_<density>_<in/out>degree_supplementary.csv` | the single example run |
| `sample_adjacency_matrix_<density>.csv` / `.png` | one sample network per density |

Here `<density>` is `sparse`, `normal` (intermediate) or `dense`.

---

## What to expect from the simulations

- Control power increases with out-degree and decreases with in-degree under every density and every (α, β) setting. The in-degree effect is largely a property of the generative model: with edges pointing from lower to higher index, vertices of high in-degree sit late in the order and have little left to reach.
- The mean score changes little with density even though the reachable sets grow a lot. In our runs the mean size of *R(v)* grew from about 8.5 (sparse) to about 23.8 (dense), while the mean score was about 11.6 (intermediate) and 11.3 (dense). The normalized distance term offsets most of the growth of |*R(v)*|.

---

## Application to cancer signaling pathways

The method was applied to five KEGG pathways after conversion to DAGs: colorectal cancer (hsa05210), glioma (hsa05214), acute myeloid leukemia (hsa05221), breast cancer (hsa05224) and gastric cancer (hsa05226). After reduction the networks have between 38 vertices and 46 edges (glioma) and 67 vertices and 70 edges (gastric cancer).

The set of five top-ranked proteins was identical under all three (α, β) settings in glioma, acute myeloid leukemia, breast cancer and gastric cancer. In colorectal cancer the only change was AREG entering the top five in place of EGFR at α = 1, β = 0.5. Functional enrichment (STRING, FDR < 0.05) and Kaplan–Meier survival analyses of these proteins are reported in Sections 3.2.2 and 3.2.3 of the paper and in the Supplementary Material. The survival associations are mixed: some top-ranked proteins are significant and others are not, and the direction of the effect differs between cancers.

---

## Notes and limitations

- **Acyclic input only.** Cycles must be removed first, for example with `dag_reduction.py`. Contracting an SCC treats all of its members as one unit, so individual members of a collapsed component do not receive separate scores. Contraction also preserves reachability but merges parallel edges between components, so path counts through collapsed regions differ from those in the original graph.
- **Topology only.** The score does not use expression data, interaction signs (activation or inhibition) or edge weights. Weights in an edge-list input are read but ignored by the scoring step.
- **Degree conventions.** In the random-DAG generator ⟨k⟩ is the mean total degree (see above). The `avg_degree` printed by `dag_reduction.py` is the number of edges divided by the number of vertices, i.e. the mean out-degree. The two are not directly comparable.
- **Prioritization, not proof.** A high score marks a structurally well-placed vertex. It is a way to shortlist candidates for follow-up, not evidence that a protein is a causal driver.

---

## Citation

If you use this code, please cite:

> Hashemi Dulabi, B. S., Kalhori, M., Afzali Nejad, S. & Ebrahimi, A. ControlPower: Bottom-Up Dynamic Programming Algorithm for Quantifying Vertex Control Strength in DAGs. (submitted)

The full reference will be added here once it is available.

## Contact

Questions and suggestions are welcome through GitHub issues. Corresponding author: Ali Ebrahimi (a.ebrahimi@alzahra.ac.ir).

## License

MIT License. See the [LICENSE](LICENSE) file for details.
