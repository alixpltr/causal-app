# methods/missdag_boot.py
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# MissDAG + MCEM init/learner (your files)
from miss_dag_nongaussian import miss_dag_nongaussian
from notears_mlp_mcem_init import Notears_MLP_MCEM_INIT
from notears_mlp_mcem import Notears_MLP_MCEM

# causal-learn graph utils for rendering
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils

import networkx as nx  # fallback draw

# ====== FIXED INTERNAL SETTINGS ======
WEIGHT_THR = 0.30   # binarize each run: edge i->j present if |B[i,j]| > WEIGHT_THR
EDGE_THR   = 0.50   # include edge in consensus if skeleton prob ≥ EDGE_THR
DIR_THR    = 0.50   # orient in consensus if conditional dir prob ≥ DIR_THR
RNG_SEED   = 12345

# MissDAG hyperparams (non-Gaussian / LiNGAM)
EM_ITER      = 5
MLE_SCORE    = "Sup-G"   # 'Sup-G' or 'Sub-G'
NUM_SAMPLING = 30
STANDARDIZE  = False     # keep NaNs; optional z-score with NaN-aware

# ---------- helpers ----------
def clean_df_keep_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric cols; convert ±inf→NaN; DO NOT drop NaNs (MissDAG handles missing)."""
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def zscore_nan(X: np.ndarray) -> np.ndarray:
    """Column-wise z-score ignoring NaNs (preserves NaNs)."""
    X = X.astype(float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd

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

    Gd = nx.DiGraph()
    Gd.add_nodes_from(labels)
    Gd.add_edges_from(dir_list)
    pos = nx.spring_layout(Gd, seed=42)

    plt.figure(figsize=(11, 8))
    nx.draw_networkx_nodes(Gd, pos, node_color="#cfe8ff", node_size=900, edgecolors="#1b4965")
    nx.draw_networkx_labels(Gd, pos, font_size=9)
    if dir_list:
        nx.draw_networkx_edges(Gd, pos, edgelist=dir_list,
                               arrows=True, arrowstyle="->", arrowsize=16, width=1.9)
    if und_list:
        Gu = nx.Graph()
        Gu.add_nodes_from(labels)
        Gu.add_edges_from(und_list)
        nx.draw_networkx_edges(Gu, pos, edgelist=und_list, style="dashed", width=1.5)

    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

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
    _ = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load (keep NaNs for MissDAG)
    df = pd.read_csv(args.data)
    df = clean_df_keep_missing(df)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns.")
    X_full = df.to_numpy(dtype=float)
    if STANDARDIZE:
        X_full = zscore_nan(X_full)
    labels = df.columns.tolist()
    n, p = X_full.shape
    label_to_idx = {c: i for i, c in enumerate(labels)}

    # ---- bootstrap counts ----
    B_eff = 0
    skel_counts = np.zeros((p, p), dtype=int)  # unordered pairs (a<b)
    dir_counts  = np.zeros((p, p), dtype=int)  # ordered arcs (i->j)

    rng = np.random.default_rng(RNG_SEED)
    print(f"[MISSDAG-BOOT] Data {X_full.shape} | n_boot={args.n_boot} | weight_thr={WEIGHT_THR}")

    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)
        Xb = X_full[rows]

        try:
            dag_init = Notears_MLP_MCEM_INIT(lambda1=0.2)
            dag_mcem = Notears_MLP_MCEM(lambda1=0.2)

            B_est, cov_est, histories = miss_dag_nongaussian(
                X=Xb,
                dag_init_method=dag_init,
                dag_method=dag_mcem,
                em_iter=EM_ITER,
                MLEScore=MLE_SCORE,
                num_sampling=NUM_SAMPLING,
                B_true=np.ones((p, p))  # dummy, not used
            )
        except Exception as e:
            print(f"[MISSDAG-BOOT] replicate {b}: FAILED -> {e}")
            continue

        # binarize this run
        A = (np.abs(B_est) > WEIGHT_THR).astype(int)
        np.fill_diagonal(A, 0)

        # skeleton present if either direction is 1
        present_pairs = set()
        for i in range(p):
            for j in range(p):
                if i == j: continue
                if A[i, j] == 1 or A[j, i] == 1:
                    a, bb = (i, j) if i < j else (j, i)
                    present_pairs.add((a, bb))
        for a, bb in present_pairs:
            skel_counts[a, bb] += 1

        # direction only if exactly one direction is present
        for i in range(p):
            for j in range(p):
                if i == j: continue
                if A[i, j] == 1 and A[j, i] == 0:
                    dir_counts[i, j] += 1
                # if both 1 (shouldn't for DAG, but if it happens) -> no direction increment

        B_eff += 1

    if B_eff == 0:
        raise RuntimeError("All MissDAG bootstraps failed; no consensus computed.")

    # ---- probabilities ----
    skel_prob = skel_counts.astype(float) / B_eff
    same_dir_cond = np.full((p, p), np.nan, dtype=float)
    for a in range(p):
        for b in range(a + 1, p):
            denom = skel_counts[a, b]
            if denom > 0:
                same_dir_cond[a, b] = dir_counts[a, b] / denom
                same_dir_cond[b, a] = dir_counts[b, a] / denom

    # ---- consensus graph (GeneralGraph) ----
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
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.TAIL))   # undirected

    # ---- save consensus PNG ----
    png_consensus = outdir / "missdag_boot_consensus.png"
    if not try_graphviz_png(consensus, labels, png_consensus):
        draw_networkx_fallback(consensus, labels, "MissDAG bootstrap consensus", png_consensus)
    print(f"[MISSDAG-BOOT] Saved: {png_consensus.name}")

    # ---- annotated PNG (skeleton%, same-dir% | pair) ----
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

        png_annot = outdir / "missdag_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[MISSDAG-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[MISSDAG-BOOT] Annotation failed: {e}")

    print(f"[MISSDAG-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[MISSDAG-BOOT] Outputs in: {outdir}")

if __name__ == "__main__":
    main()
