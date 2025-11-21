# methods/grandag_boot.py
# -*- coding: utf-8 -*-

import os
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict, Counter

# GraNDAG (gCastle)
os.environ.setdefault("CASTLE_BACKEND", "pytorch")
from castle.algorithms import GraNDAG

# fallback drawing
import networkx as nx

# causallearn / graphviz helpers
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils

# ----------------- config (fixed; thresholds can be changed here) -----------------
EDGE_THR = 0.50   # include pair if skeleton prob >= this
DIR_THR  = 0.50   # direct an edge if conditional dir prob >= this (and >= opposite)
RNG_SEED = 12345  # bootstrap RNG seed

# ----------------- data cleaning -----------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return df

# ----------------- graphviz / pydot helpers -----------------
def try_graphviz_png(cg_or_graph, labels, out_png: pathlib.Path) -> bool:
    """
    Use Graphviz via pydot for clean layout. Return True if success.
    """
    try:
        pyd = GraphUtils.to_pydot(
            cg_or_graph.G if hasattr(cg_or_graph, "G") else cg_or_graph,
            labels=labels
        )
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pyd.write_png(str(out_png))
        return True
    except Exception:
        return False

def pydot_label_map(pyd):
    """Map displayed node label -> internal node id (strings)."""
    m = {}
    for n in pyd.get_nodes():
        node_id = str(n.get_name()).strip('"').strip()
        label = n.get("label")
        label = node_id if label is None else str(label).strip('"').strip()
        m[label] = node_id
    return m

def set_edge_xlabel(pyd, src_id, dst_id, text):
    """Attach external label to edge (robust lookup)."""
    edges = pyd.get_edge(src_id, dst_id) or pyd.get_edge(f'"{src_id}"', f'"{dst_id}"') or []
    if not edges:
        for e in pyd.get_edges():
            s = str(e.get_source()).strip('"').strip()
            t = str(e.get_destination()).strip('"').strip()
            if s == src_id and t == dst_id:
                edges = [e]
                break
    for e in edges:
        e.set("xlabel", text)
        e.set("labelfontsize", "10")
        e.set("labelfontcolor", "black")
        e.set("labeldistance", "1.6")
        e.set("labelangle", "0")

# ----------------- run a single GraNDAG fit and return binary adjacency -----------------
def run_grandag_binary_adj(X_std: np.ndarray, labels: list[str]) -> np.ndarray:
    """
    Train GraNDAG once on standardized data and return a 0/1 adjacency (rows -> cols).
    Uses your exact parameter set from the prompt.
    """
    d = X_std.shape[1]
    grandag_kwargs = dict(
        input_dim=d,
        hidden_num=3,
        hidden_dim=64,        # as requested
        batch_size=64,
        lr=1e-3,
        iterations=1000,
        model_name='NonLinGaussANM',
        nonlinear='leaky-relu',
        optimizer='rmsprop',
        h_threshold=5e-4,
        device_type='cpu',
        device_ids='0',
        use_pns=True,         # as in your snippet
        pns_thresh=0.75,
        num_neighbors=15,
        normalize=False,      # we standardize outside
        precision=False,
        random_seed=42,
        jac_thresh=True,      # as in your snippet
        lambda_init=0.0,
        mu_init=1e-4,
        omega_lambda=1e-4,
        omega_mu=0.7,
        stop_crit_win=100,
        edge_clamp_range=0.0,
        norm_prod='paths',
        square_prod=False
    )

    np.random.seed(grandag_kwargs["random_seed"])
    gnd = GraNDAG(**grandag_kwargs)
    gnd.learn(data=X_std, columns=labels)

    # prefer causal_matrix if present; otherwise try model.adjacency
    if hasattr(gnd, "causal_matrix") and gnd.causal_matrix is not None:
        A = np.asarray(gnd.causal_matrix, dtype=int)
    elif hasattr(gnd, "model") and hasattr(gnd.model, "adjacency"):
        A = np.asarray(gnd.model.adjacency.detach().cpu().numpy(), dtype=int)
    else:
        raise AttributeError("No adjacency found on GraNDAG object.")
    return A

# ----------------- consensus helpers -----------------
def build_consensus_graph(labels, skel_prob, same_dir_cond, edge_thr=EDGE_THR, dir_thr=DIR_THR):
    """
    Build a causallearn GeneralGraph consensus:
    - include pair if skeleton prob >= edge_thr
    - orient a->b if P(a->b | pair) >= dir_thr and >= opposite
      else add as undirected (tail-tail)
    """
    p = len(labels)
    nodes = [GraphNode(lbl) for lbl in labels]
    Gc = GeneralGraph(nodes)

    for a in range(p):
        for b in range(a+1, p):
            sp = skel_prob[a, b]
            if sp < edge_thr:
                continue
            pab = 0.0 if np.isnan(same_dir_cond[a, b]) else same_dir_cond[a, b]
            pba = 0.0 if np.isnan(same_dir_cond[b, a]) else same_dir_cond[b, a]

            if pab >= pba and pab >= dir_thr:
                Gc.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.ARROW))
            elif pba > pab and pba >= dir_thr:
                Gc.add_edge(Edge(nodes[b], nodes[a], Endpoint.TAIL, Endpoint.ARROW))
            else:
                Gc.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.TAIL))

    return Gc

def draw_fallback_png(consensus_graph: GeneralGraph, labels, png_path: pathlib.Path):
    """Fallback drawing with networkx if Graphviz isn't available."""
    dir_list, und_list = [], []
    for e in consensus_graph.get_graph_edges():
        s = e.get_node1().get_name()
        t = e.get_node2().get_name()
        ep1, ep2 = e.get_endpoint1().value, e.get_endpoint2().value
        if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
            dir_list.append((s, t))
        elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
            und_list.append((s, t))
    Gd = nx.DiGraph()
    Gd.add_nodes_from(labels)
    Gd.add_edges_from(dir_list)
    pos = nx.spring_layout(Gd, seed=42)
    plt.figure(figsize=(11, 8))
    nx.draw_networkx_nodes(Gd, pos, node_color="#E6F0FF", node_size=900, edgecolors="#345")
    nx.draw_networkx_labels(Gd, pos, font_size=9)
    if dir_list:
        nx.draw_networkx_edges(Gd, pos, edgelist=dir_list, arrows=True, arrowstyle="->", arrowsize=16, width=1.9)
    if und_list:
        Gu = nx.Graph(); Gu.add_nodes_from(labels); Gu.add_edges_from(und_list)
        nx.draw_networkx_edges(Gu, pos, edgelist=und_list, style="dashed", width=1.5)
    plt.title("GraNDAG bootstrap consensus (fallback)")
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=150)
    plt.close()

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input CSV")
    ap.add_argument("--outdir", required=True, help="Directory to save outputs")
    ap.add_argument("--n_boot", type=int, default=20, help="Bootstrap replicates")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # load & clean
    df = pd.read_csv(args.data)
    df = clean_df(df)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns after cleaning.")
    labels = df.columns.tolist()
    X_full = df.to_numpy(dtype=float)
    n, p = X_full.shape

    # standardize once; bootstrap resamples from standardized rows
    mu = X_full.mean(axis=0)
    sigma = X_full.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    X_std = (X_full - mu) / sigma

    print(f"[GraNDAG-BOOT] Data {X_full.shape} | n_boot={args.n_boot}")

    # tallies
    B_eff = 0
    skel_counts = np.zeros((p, p), dtype=int)  # unordered: a<b
    dir_counts  = np.zeros((p, p), dtype=int)  # ordered: i->j (directed)
    rng = np.random.default_rng(RNG_SEED)

    # bootstrap
    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)
        Xb = X_std[rows, :]
        try:
            A = run_grandag_binary_adj(Xb, labels)  # 0/1 adjacency
        except Exception as e:
            print(f"[GraNDAG-BOOT] replicate {b}: FAILED -> {e}")
            continue

        # counts
        for i in range(p):
            for j in range(i+1, p):
                a, bb = i, j
                if A[i, j] == 1 or A[j, i] == 1:
                    skel_counts[a, bb] += 1
                if A[i, j] == 1:
                    dir_counts[i, j] += 1
                if A[j, i] == 1:
                    dir_counts[j, i] += 1

        B_eff += 1

    if B_eff == 0:
        raise RuntimeError("All GraNDAG bootstraps failed; no consensus computed.")

    # probabilities
    skel_prob = skel_counts.astype(float) / B_eff
    same_dir_cond = np.full((p, p), np.nan, dtype=float)  # P(i->j | pair)
    for a in range(p):
        for b in range(a+1, p):
            denom = skel_counts[a, b]
            if denom > 0:
                same_dir_cond[a, b] = dir_counts[a, b] / denom
                same_dir_cond[b, a] = dir_counts[b, a] / denom

    # consensus graph
    consensus = build_consensus_graph(labels, skel_prob, same_dir_cond,
                                      edge_thr=EDGE_THR, dir_thr=DIR_THR)

    # save PNG (plain)
    png_consensus = outdir / "grandag_boot_consensus.png"
    if try_graphviz_png(consensus, labels, png_consensus):
        print(f"[GraNDAG-BOOT] Saved: {png_consensus.name}")
    else:
        draw_fallback_png(consensus, labels, png_consensus)
        print(f"[GraNDAG-BOOT] Saved (fallback): {png_consensus.name}")

    # annotated PNG
    try:
        pyd = GraphUtils.to_pydot(consensus, labels=labels)
        label2id = pydot_label_map(pyd)

        for e in consensus.get_graph_edges():
            n1 = e.get_node1().get_name()
            n2 = e.get_node2().get_name()
            i = labels.index(n1); j = labels.index(n2)
            a, b = (i, j) if i < j else (j, i)

            sk = skel_prob[a, b]  # in [0,1]
            ep1 = e.get_endpoint1().value
            ep2 = e.get_endpoint2().value

            if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
                # i -> j
                pd_dir = same_dir_cond[i, j]
                dir_txt = "—" if np.isnan(pd_dir) else f"{pd_dir*100:.0f}%"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, {dir_txt})")
            elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
                # undirected => only skeleton %
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, —)")

        png_annot = outdir / "grandag_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[GraNDAG-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[GraNDAG-BOOT] Annotation failed: {e}")

    # save arrays
    np.save(outdir / "grandag_boot_skeleton_prob.npy", skel_prob)
    np.save(outdir / "grandag_boot_dir_prob.npy", same_dir_cond)

    print(f"[GraNDAG-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[GraNDAG-BOOT] Outputs in: {outdir}")

if __name__ == "__main__":
    main()
