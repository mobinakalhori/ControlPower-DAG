from __future__ import annotations
import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd

__all__ = [
    "GraphStats",
    "load_adjacency_matrix",
    "load_edge_list",
    "remove_isolated_nodes",
    "condense_to_dag",
    "compute_graph_stats",
    "save_adjacency_list",
    "save_adjacency_matrix",
    "reduce_to_dag",
    "build_dag_from_matrix_file",
    "build_dag_from_edge_list_file",
]

LOGGER = logging.getLogger("dag_reduction")

@dataclass
class GraphStats:
    """Basic summary statistics for a directed graph."""

    n_nodes: int
    n_edges: int
    avg_degree: float
    diameter: int  # longest path length; only meaningful when the graph is a DAG
    is_dag: bool

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"n={self.n_nodes}, m={self.n_edges}, "
            f"<d>={self.avg_degree:.2f}, diam={self.diameter}, "
            f"is_dag={self.is_dag}"
        )

def load_adjacency_matrix(path: str, sep: str = "\t") -> nx.DiGraph:
    """
    Load a directed graph from a square adjacency-matrix file.

    The file is expected to be tab-separated, with node names in the first
    row and first column. Entry (i, j) gives the weight of the edge from
    i to j (zero means no edge).

    Parameters
    ----------
    path : str
        Path to the adjacency-matrix file.
    sep : str
        Field separator (default: tab).

    Returns
    -------
    networkx.DiGraph
        Directed graph using the original node names as labels.
    """
    df = pd.read_table(path, sep=sep, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"Adjacency matrix in '{path}' is not square: "
            f"{df.shape[0]} rows vs {df.shape[1]} columns."
        )
    df.index = df.index.map(str)
    df.columns = df.columns.map(str)
    if list(df.index) != list(df.columns):
        LOGGER.warning(
            "Row and column labels of '%s' do not match exactly; "
            "aligning columns to the row order.",
            path,
        )
        df = df.reindex(columns=df.index)

    G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    LOGGER.info(
        "Loaded adjacency matrix '%s': %d nodes, %d edges.",
        path, G.number_of_nodes(), G.number_of_edges(),
    )
    return G

def load_edge_list(path: str, sep: str = "\t", has_header: bool = False) -> nx.DiGraph:
    """
    Load a directed graph from a two- or three-column edge list
    (source, target[, weight]).

    Parameters
    ----------
    path : str
        Path to the edge-list file.
    sep : str
        Field separator.
    has_header : bool
        Whether the file starts with a header row.

    Returns
    -------
    networkx.DiGraph
    """
    df = pd.read_table(path, sep=sep, header=0 if has_header else None)
    if df.shape[1] < 2:
        raise ValueError(
            f"Edge list '{path}' must have at least 2 columns (source, target)."
        )

    G = nx.DiGraph()
    for row in df.itertuples(index=False):
        u, v = str(row[0]), str(row[1])
        w = float(row[2]) if len(row) > 2 and pd.notna(row[2]) else 1.0
        G.add_edge(u, v, weight=w)

    LOGGER.info(
        "Loaded edge list '%s': %d nodes, %d edges.",
        path, G.number_of_nodes(), G.number_of_edges(),
    )
    return G

def save_adjacency_list(G: nx.DiGraph, path: str) -> None:
    """
    Write the graph as a plain-text adjacency list.

    Each line has the form:

        node_name<TAB>succ_1;succ_2;...;succ_k

    Nodes with no outgoing edges are still listed, with an empty
    successor field, so every node appears in the output.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for node in G.nodes():
            successors = list(G.successors(node))
            f.write(f"{node}\t{';'.join(successors)}\n")
    LOGGER.info("Adjacency list written to '%s' (%d nodes).", path, G.number_of_nodes())

def save_adjacency_matrix(G: nx.DiGraph, path: str, sep: str = "\t") -> None:
    """
    Write the graph as a labelled adjacency matrix (TSV).

    Rows and columns are named by the node labels. The resulting file
    can be used directly as input for the control-power calculation.
    """
    nodes = list(G.nodes())
    df = nx.to_pandas_adjacency(G, nodelist=nodes, dtype=int)
    df.to_csv(path, sep=sep)
    LOGGER.info("Adjacency matrix written to '%s' (%d x %d).", path, len(nodes), len(nodes))

def remove_isolated_nodes(G: nx.DiGraph) -> nx.DiGraph:
    """
    Drop nodes that have neither incoming nor outgoing edges.

    Isolated vertices cannot exert or receive control, so they are
    removed before the control-power scores are computed. The input
    graph is left unchanged; a copy is returned.
    """
    isolated = [
        n for n in G.nodes()
        if G.in_degree(n) == 0 and G.out_degree(n) == 0
    ]
    if not isolated:
        return G.copy()

    LOGGER.info("Removing %d isolated node(s).", len(isolated))
    H = G.copy()
    H.remove_nodes_from(isolated)
    return H

def condense_to_dag(
    G: nx.DiGraph, name_sep: str = ";"
) -> Tuple[nx.DiGraph, Dict[str, List[str]]]:
    """
    Turn an arbitrary directed graph into a DAG by collapsing its
    strongly connected components.

    Any SCC that contains more than one node corresponds to at least one
    directed cycle. Replacing each such component with a single
    super-node removes every cycle while keeping the connectivity
    between different components. The result is guaranteed to be acyclic.

    Parameters
    ----------
    G : networkx.DiGraph
        Input digraph (may contain cycles).
    name_sep : str
        String used to join the original node names inside a super-node
        (default: ";").

    Returns
    -------
    dag : networkx.DiGraph
        Acyclic graph. Super-nodes are labelled by the joined names of
        their members; nodes that did not belong to any cycle keep
        their original labels.
    mapping : dict[str, list[str]]
        For each node in the DAG, the list of original vertices it
        represents.
    """
    if G.number_of_nodes() == 0:
        return G.copy(), {}

    if nx.is_directed_acyclic_graph(G):
        LOGGER.info("Input graph is already acyclic; no condensation needed.")
        dag = nx.DiGraph()
        dag.add_nodes_from(G.nodes())
        dag.add_edges_from(G.edges())
        mapping = {n: [n] for n in G.nodes()}
        return dag, mapping

    LOGGER.info("Input graph contains cycles; computing SCC condensation.")
    sccs = list(nx.strongly_connected_components(G))
    n_nontrivial = sum(1 for c in sccs if len(c) > 1)
    LOGGER.info(
        "Found %d strongly connected component(s), %d of which contain a cycle.",
        len(sccs),
        n_nontrivial,
    )

    # condensation() collapses each SCC into one node and always returns a DAG
    cond = nx.condensation(G, scc=sccs)

    relabel: Dict[int, str] = {}
    mapping: Dict[str, List[str]] = {}
    for idx in cond.nodes():
        members = sorted(cond.nodes[idx]["members"])
        new_name = name_sep.join(members)
        relabel[idx] = new_name
        mapping[new_name] = members

    dag = nx.relabel_nodes(cond, relabel)
    # safety: strip any self-loops that might appear on malformed input
    dag.remove_edges_from(list(nx.selfloop_edges(dag)))

    assert nx.is_directed_acyclic_graph(dag), "Condensation failed to produce a DAG."
    return dag, mapping

def compute_graph_stats(G: nx.DiGraph) -> GraphStats:
    """
    Compute a short summary of the graph: number of nodes and edges,
    average degree, diameter (longest path length), and whether the
    graph is acyclic.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_degree = m / n if n > 0 else 0.0
    is_dag = nx.is_directed_acyclic_graph(G)
    if is_dag and n > 0:
        diameter = nx.dag_longest_path_length(G)
    else:
        # diameter is only well-defined once the graph is a DAG
        diameter = -1
    return GraphStats(
        n_nodes=n,
        n_edges=m,
        avg_degree=avg_degree,
        diameter=diameter,
        is_dag=is_dag,
    )


def reduce_to_dag(
    G: nx.DiGraph,
    drop_isolated: bool = True,
    name_sep: str = ";",
) -> Tuple[nx.DiGraph, GraphStats, Dict[str, List[str]]]:
    """
    Full conversion of a directed graph into a DAG.

    Steps:
        1. optionally remove isolated nodes;
        2. collapse strongly connected components so that every cycle
           disappears;
        3. optionally remove any isolated nodes that condensation may
           have created.

    The resulting DAG is the structure on which control-power scores
    are subsequently computed.

    Parameters
    ----------
    G : networkx.DiGraph
        Input digraph.
    drop_isolated : bool
        Whether to discard isolated nodes before and after condensation.
    name_sep : str
        Separator used when naming super-nodes.

    Returns
    -------
    dag : networkx.DiGraph
        The acyclic graph.
    stats : GraphStats
        Summary statistics of the DAG.
    mapping : dict[str, list[str]]
        Mapping from each DAG node to the original vertices it stands for.
    """
    H = remove_isolated_nodes(G) if drop_isolated else G.copy()

    dag, mapping = condense_to_dag(H, name_sep=name_sep)

    if drop_isolated:
        before = dag.number_of_nodes()
        dag = remove_isolated_nodes(dag)
        after = dag.number_of_nodes()
        if after < before:
            LOGGER.info(
                "Removed %d additional isolated node(s) created by condensation.",
                before - after,
            )
        mapping = {k: v for k, v in mapping.items() if k in dag.nodes()}

    stats = compute_graph_stats(dag)
    LOGGER.info("Final DAG statistics: %s", stats)
    return dag, stats, mapping


def build_dag_from_matrix_file(
    path: str,
    sep: str = "\t",
    drop_isolated: bool = True,
    name_sep: str = ";",
) -> Tuple[nx.DiGraph, GraphStats, Dict[str, List[str]]]:
    """Load an adjacency-matrix file and reduce it to a DAG."""
    G = load_adjacency_matrix(path, sep=sep)
    return reduce_to_dag(G, drop_isolated=drop_isolated, name_sep=name_sep)


def build_dag_from_edge_list_file(
    path: str,
    sep: str = "\t",
    has_header: bool = False,
    drop_isolated: bool = True,
    name_sep: str = ";",
) -> Tuple[nx.DiGraph, GraphStats, Dict[str, List[str]]]:
    """Load an edge-list file and reduce it to a DAG."""
    G = load_edge_list(path, sep=sep, has_header=has_header)
    return reduce_to_dag(G, drop_isolated=drop_isolated, name_sep=name_sep)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a directed graph into a DAG by removing isolated "
            "nodes and collapsing strongly connected components. "
            "The resulting DAG is used as input for control-power ranking."
        )
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to the input graph file."
    )
    parser.add_argument(
        "--input-format",
        "-f",
        choices=["matrix", "edgelist"],
        default="matrix",
        help="Input format: 'matrix' (labelled square adjacency matrix) "
             "or 'edgelist' (source, target[, weight]). Default: matrix.",
    )
    parser.add_argument(
        "--sep", default="\t",
        help="Field separator for input/output files. Default: tab.",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Set this flag if the edge-list file has a header row.",
    )
    parser.add_argument(
        "--keep-isolated",
        action="store_true",
        help="Keep isolated nodes (by default they are removed).",
    )
    parser.add_argument(
        "--name-sep",
        default=";",
        help="Separator used when joining names of collapsed nodes. Default: ';'.",
    )
    parser.add_argument(
        "--output-list",
        "-ol",
        default=None,
        help="Write the DAG as an adjacency list (node<TAB>succ1;succ2;...).",
    )
    parser.add_argument(
        "--output-matrix",
        "-om",
        default=None,
        help="Write the DAG as a labelled adjacency matrix (TSV).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    if args.output_list is None and args.output_matrix is None:
        parser.error("At least one of --output-list / --output-matrix must be given.")

    try:
        if args.input_format == "matrix":
            G = load_adjacency_matrix(args.input, sep=args.sep)
        else:
            G = load_edge_list(args.input, sep=args.sep, has_header=args.has_header)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("Could not load input graph: %s", exc)
        return 1

    dag, stats, mapping = reduce_to_dag(
        G, drop_isolated=not args.keep_isolated, name_sep=args.name_sep
    )

    n_collapsed = sum(1 for members in mapping.values() if len(members) > 1)
    print("=" * 60)
    print(
        f"Input graph : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
        f"acyclic={nx.is_directed_acyclic_graph(G)}"
    )
    print(f"Output DAG  : {stats}")
    print(f"Super-nodes created by cycle removal: {n_collapsed}")
    if n_collapsed:
        print("Collapsed groups (original nodes merged into one DAG node):")
        for name, members in mapping.items():
            if len(members) > 1:
                print(f"  - {name}  <=  {members}")
    print("=" * 60)

    if args.output_list:
        save_adjacency_list(dag, args.output_list)
        print(f"Adjacency list written to: {args.output_list}")
    if args.output_matrix:
        save_adjacency_matrix(dag, args.output_matrix, sep=args.sep)
        print(f"Adjacency matrix written to: {args.output_matrix}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
