# methods/notears_sob_boot.py
# -*- coding: utf-8 -*-

import os
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# gCastle
os.environ.setdefault("CASTLE_BACKEND", "pytorch")
try:
    from castle.algorithms import NotearsNonlinear
except Exception as e:
    raise ImportError("Install gCastle: pip install gcastle") from e

# fallback draw
import networkx as nx

# causallearn / graphviz helpers
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils

# ----------------- fixed thresholds (tune here if needed) -----------------
EDGE_THR = 0.50   # include pair if skeleton prob >= this
DIR_THR  = 0.50   # orient if P(i->j | pair) >= this AND >= opposite
RNG_SEED = 12345  # bootstrap RNG seed

# ----------------- utilities -----------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return df

def to_dataframe(mat, idx, cols):
    if isinstance(mat, pd.DataFrame):
        return mat
    try:
        return pd.DataFrame(mat, index=idx, columns=cols)
    except Exception:
        return pd.DataFrame(np.asarray(mat), index=idx, columns=cols)

def try_graphviz_png(cg_or_graph, labels, out_png: pathlib.Path) -> bool:
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
    m = {}
    for n in pyd.get_nodes():
        node_id = str(n.get_name()).strip('"').strip()
        label = n.get("label")
        label = node_id if label is None else str(label).strip('"').strip()
        m[label] = node_id
    return m

def set_edge_xlabel(pyd, src_id, dst_id, text):
    edges = pyd.get_edge(src_id, dst_id) or pyd.get_edge(f'"{src_id}"', f'"{dst_id}"') or []
    if not edges:
        for e in pyd.get_edges():
            s = str(e.get_source()).strip('"').strip()
            t = str(e.get_destination()).strip('"').strip()
            if s == src_id and t == dst_id:
                edges = [e]; break
    for e in edges:
        e.set("xlabel", text)
        e.set("labelfontsize", "10")
        e.set("labelfontcolor", "black")
        e.set("labeldistance", "1.6")
        e.set("labelangle", "0")

def draw_fallback_png(consensus_graph: GeneralGraph, labels, png_path: pathlib.Path):
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
    plt.title("NOTEARS-SOB bootstrap consensus (fallback)")
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=150)
    plt.close()

# ----------------- single run -----------------
def run_notears_sob_binary_adj(X_std: np.ndarray, labels: list[str], w_thr: float) -> np.ndarray:
    """
    Train NOTEARS-SOB once on standardized data and return a 0/1 adjacency (rows -> cols).
    """
    hparams = dict(
        lambda1=0.01,
        lambda2=0.01,
        max_iter=100,
        h_tol=1e-6,
        rho_max=1e16,
        w_threshold=w_thr,     # only used if causal_matrix not provided
        expansions=15,
        bias=True,
        model_type="sob",
        device_type="cpu",
        device_ids=None,
    )

    np.random.seed(42)
    sob = NotearsNonlinear(**hparams)
    sob.learn(X_std, columns=labels)

    if hasattr(sob, "causal_matrix") and sob.causal_matrix is not None:
        A = np.asarray(sob.causal_matrix, dtype=int)
    else:
        if not hasattr(sob, "weight_causal_matrix"):
            raise AttributeError("NotearsNonlinear has no 'weight_causal_matrix'. Update gCastle.")
        W = np.asarray(sob.weight_causal_matrix, dtype=float)
        A = (np.abs(W) > w_thr).astype(int)
    return A

# ----------------- consensus build -----------------
def build_consensus_graph(labels, skel_prob, same_dir_cond, edge_thr=EDGE_THR, dir_thr=DIR_THR):
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

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input CSV")
    ap.add_argument("--outdir", required=True, help="Directory to save outputs")
    ap.add_argument("--n_boot", type=int, default=20, help="Bootstrap replicates")
    ap.add_argument("--w_thr", type=float, default=0.40, help="|weight| threshold if binarizing")
    ap.add_argument("--edge_thr", type=float, default=EDGE_THR, help="Skeleton prob threshold")
    ap.add_argument("--dir_thr", type=float, default=DIR_THR, help="Direction prob threshold")
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

    # standardize once; resample rows from standardized matrix
    mu = X_full.mean(axis=0)
    sigma = X_full.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    X_std = (X_full - mu) / sigma

    print(f"[NOTEARS-SOB-BOOT] Data {X_full.shape} | n_boot={args.n_boot}")

    # tallies
    B_eff = 0
    skel_counts = np.zeros((p, p), dtype=int)  # unordered: a<b
    dir_counts  = np.zeros((p, p), dtype=int)  # ordered: i->j
    rng = np.random.default_rng(RNG_SEED)

    # bootstrap
    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)
        Xb = X_std[rows, :]
        try:
            A = run_notears_sob_binary_adj(Xb, labels, w_thr=args.w_thr)
        except Exception as e:
            print(f"[NOTEARS-SOB-BOOT] replicate {b}: FAILED -> {e}")
            continue

        for i in range(p):
            for j in range(i+1, p):
                if A[i, j] == 1 or A[j, i] == 1:
                    skel_counts[i, j] += 1
                if A[i, j] == 1:
                    dir_counts[i, j] += 1
                if A[j, i] == 1:
                    dir_counts[j, i] += 1

        B_eff += 1

    if B_eff == 0:
        raise RuntimeError("All NOTEARS-SOB bootstraps failed; no consensus computed.")

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
                                      edge_thr=args.edge_thr, dir_thr=args.dir_thr)

    # save PNG (plain)
    png_consensus = outdir / "notears_sob_boot_consensus.png"
    if try_graphviz_png(consensus, labels, png_consensus):
        print(f"[NOTEARS-SOB-BOOT] Saved: {png_consensus.name}")
    else:
        draw_fallback_png(consensus, labels, png_consensus)
        print(f"[NOTEARS-SOB-BOOT] Saved (fallback): {png_consensus.name}")

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
                pd_dir = same_dir_cond[i, j]
                dir_txt = "—" if np.isnan(pd_dir) else f"{pd_dir*100:.0f}%"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, {dir_txt})")
            elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, —)")

        png_annot = outdir / "notears_sob_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[NOTEARS-SOB-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[NOTEARS-SOB-BOOT] Annotation failed: {e}")

    # save arrays
    np.save(outdir / "notears_sob_boot_skeleton_prob.npy", skel_prob)
    np.save(outdir / "notears_sob_boot_dir_prob.npy", same_dir_cond)

    print(f"[NOTEARS-SOB-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[NOTEARS-SOB-BOOT] Outputs in: {outdir}")

if __name__ == "__main__":
    main()
