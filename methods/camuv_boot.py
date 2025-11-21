# methods/camuv_boot.py
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from causallearn.search.FCMBased.lingam import CAMUV

# causal-learn graph utils for rendering
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils

import networkx as nx  # fallback draw

# ===== fixed internal settings (no CLI exposure) =====
ALPHA = 0.01
NUM_EXPL_VALUES = 5

EDGE_THR = 0.50   # include pair in consensus if skeleton prob ≥ this
DIR_THR  = 0.50   # orient if conditional same-dir prob ≥ this (and ≥ opposite)
RNG_SEED = 12345

# ---------- helpers ----------
def clean_df_no_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric-only; convert ±inf→NaN; drop NaN rows (CAM-UV expects full rows)."""
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return df

def standardize(df: pd.DataFrame) -> np.ndarray:
    """z-score columns."""
    X = df.to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)
    return X

def try_graphviz_png(GG: GeneralGraph, labels, out_png: pathlib.Path) -> bool:
    try:
        pyd = GraphUtils.to_pydot(GG, labels=labels)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pyd.write_png(str(out_png))
        return True
    except Exception:
        return False

def draw_networkx_fallback(consensus: GeneralGraph, labels, title, out_png: pathlib.Path):
    dir_list, und_list = [], []
    for e in consensus.get_graph_edges():
        s = e.get_node1().get_name()
        t = e.get_node2().get_name()
        ep1, ep2 = e.get_endpoint1().value, e.get_endpoint2().value
        if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
            dir_list.append((s, t))
        elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
            und_list.append((s, t))

    Gd = nx.DiGraph(); Gd.add_nodes_from(labels); Gd.add_edges_from(dir_list)
    pos = nx.spring_layout(Gd, seed=42)

    plt.figure(figsize=(11, 8))
    nx.draw_networkx_nodes(Gd, pos, node_color="#cfe8ff", node_size=900, edgecolors="#1b4965")
    nx.draw_networkx_labels(Gd, pos, font_size=9)
    if dir_list:
        nx.draw_networkx_edges(Gd, pos, edgelist=dir_list, arrows=True, arrowstyle="->",
                               arrowsize=16, width=1.9)
    if und_list:
        Gu = nx.Graph(); Gu.add_nodes_from(labels); Gu.add_edges_from(und_list)
        nx.draw_networkx_edges(Gu, pos, edgelist=und_list, style="dashed", width=1.5)

    plt.title(title); plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()

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

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input CSV")
    ap.add_argument("--outdir", required=True, help="Directory to save outputs")
    ap.add_argument("--n_boot", type=int, default=20, help="Bootstrap replicates")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # load, clean, standardize
    df = pd.read_csv(args.data)
    df = clean_df_no_missing(df)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns after cleaning.")
    labels = df.columns.tolist()
    X_full = standardize(df)
    n, p = X_full.shape
    print(f"[CAMUV-BOOT] Data {X_full.shape} | n_boot={args.n_boot} | alpha={ALPHA}, K={NUM_EXPL_VALUES}")

    # ---------- bootstrap counters ----------
    B_eff = 0
    skel_counts = np.zeros((p, p), dtype=int)  # unordered pairs (a<b): any of ->, <-, or U present
    dir_counts  = np.zeros((p, p), dtype=int)  # ordered arcs (i->j)

    rng = np.random.default_rng(RNG_SEED)

    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)
        Xb = X_full[rows, :]

        try:
            # CAM-UV returns parents list P (per node) and ambiguous list U (pairs)
            P, U = CAMUV.execute(Xb, alpha=ALPHA, num_explanatory_vals=NUM_EXPL_VALUES)
        except Exception as e:
            print(f"[CAMUV-BOOT] replicate {b}: FAILED -> {e}")
            continue

        # Build directed adjacency from parents: p -> i
        A_dir = np.zeros((p, p), dtype=int)
        for i, parents in enumerate(P):
            for par in parents:
                if 0 <= par < p and par != i:
                    A_dir[par, i] = 1

        # ----- skeleton counting -----
        # skeleton present if directed either way, or listed in U
        present_pairs = set()
        # from directed
        for i in range(p):
            for j in range(p):
                if i == j: continue
                if A_dir[i, j] == 1 or A_dir[j, i] == 1:
                    a, bb = (i, j) if i < j else (j, i)
                    present_pairs.add((a, bb))
        # from ambiguous U (treat as undirected adjacency)
        for (i, j) in U:
            if i == j: continue
            if 0 <= i < p and 0 <= j < p:
                a, bb = (i, j) if i < j else (j, i)
                present_pairs.add((a, bb))

        for a, bb in present_pairs:
            skel_counts[a, bb] += 1

        # direction counts (only when exactly one direction present)
        for i in range(p):
            for j in range(p):
                if i == j: continue
                if A_dir[i, j] == 1 and A_dir[j, i] == 0:
                    dir_counts[i, j] += 1

        B_eff += 1

    if B_eff == 0:
        raise RuntimeError("All CAM-UV bootstraps failed; no consensus computed.")

    # ---------- probabilities ----------
    skel_prob = skel_counts.astype(float) / B_eff
    same_dir_cond = np.full((p, p), np.nan, dtype=float)  # P(i->j | pair)
    for a in range(p):
        for b in range(a + 1, p):
            denom = skel_counts[a, b]
            if denom > 0:
                same_dir_cond[a, b] = dir_counts[a, b] / denom
                same_dir_cond[b, a] = dir_counts[b, a] / denom

    # ---------- consensus graph ----------
    nodes = [GraphNode(lbl) for lbl in labels]
    consensus = GeneralGraph(nodes)

    for a in range(p):
        for b in range(a + 1, p):
            sp = skel_prob[a, b]
            if sp < EDGE_THR:
                continue
            pab = 0.0 if np.isnan(same_dir_cond[a, b]) else same_dir_cond[a, b]
            pba = 0.0 if np.isnan(same_dir_cond[b, a]) else same_dir_cond[b, a]

            if pab >= pba and pab >= DIR_THR:
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.ARROW))  # a->b
            elif pba > pab and pba >= DIR_THR:
                consensus.add_edge(Edge(nodes[b], nodes[a], Endpoint.TAIL, Endpoint.ARROW))  # b->a
            else:
                # leave undirected when neither direction is confident
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.TAIL))

    # ---------- save consensus PNG ----------
    png_consensus = outdir / "camuv_boot_consensus.png"
    if not try_graphviz_png(consensus, labels, png_consensus):
        draw_networkx_fallback(consensus, labels, "CAM-UV bootstrap consensus", png_consensus)
    print(f"[CAMUV-BOOT] Saved: {png_consensus.name}")

    # ---------- annotated PNG (skeleton%, same-dir% | pair) ----------
    try:
        pyd = GraphUtils.to_pydot(consensus, labels=labels)
        label2id = pydot_label_map(pyd)

        for e in consensus.get_graph_edges():
            n1 = e.get_node1().get_name()
            n2 = e.get_node2().get_name()
            i, j = labels.index(n1), labels.index(n2)
            a, b = (i, j) if i < j else (j, i)

            sk = skel_prob[a, b]
            ep1, ep2 = e.get_endpoint1().value, e.get_endpoint2().value

            if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
                sd = same_dir_cond[i, j]
                same_txt = "—" if np.isnan(sd) else f"{sd*100:.0f}%"
                text = f"({sk*100:.0f}%, {same_txt})"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], text)
            elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
                text = f"({sk*100:.0f}%, —)"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], text)

        png_annot = outdir / "camuv_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[CAMUV-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[CAMUV-BOOT] Annotation failed: {e}")

    # optional: save raw arrays
    np.save(outdir / "camuv_boot_skeleton_prob.npy", skel_prob)
    np.save(outdir / "camuv_boot_dir_prob.npy", same_dir_cond)

    print(f"[CAMUV-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[CAMUV-BOOT] Outputs in: {outdir}")

if __name__ == "__main__":
    main()
