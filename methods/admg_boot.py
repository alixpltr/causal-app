# methods/admg_boot.py
import argparse
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime

# your local discovery classes
from run_discovery import ADMG, Discovery  # must be importable from this folder

# draw via causal-learn
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils

import matplotlib.pyplot as plt
import networkx as nx  # fallback draw

# ====== FIXED INTERNAL SETTINGS (no CLI exposure) ======
EDGE_THR = 0.50   # include pair in consensus if skeleton prob ≥ this
DIR_THR  = 0.50   # orient to a->b if P(a->b | pair) ≥ this and ≥ P(b->a | pair)
BI_THR   = 0.50   # orient to bidirected (a<->b) if P(<-> | pair) ≥ this and ≥ both directions
RNG_SEED = 12345

# Discovery hyperparams (keep here so app only passes data/outdir/n_boot)
ADMG_CLASS   = "bowfree"   # "ancestral" | "arid" | "bowfree"
LAMBDA       = 0.07
NUM_RESTARTS = 3
W_THRESHOLD  = 0.05
VERBOSE_FULL = False
VERBOSE_BOOT = False

# ---------- helpers ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric-only, drop rows with NaN/Inf (adjust if your implementation handles NaNs)."""
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return df

def try_graphviz_png(GG: GeneralGraph, labels, out_png: pathlib.Path) -> bool:
    try:
        pyd = GraphUtils.to_pydot(GG, labels=labels)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pyd.write_png(str(out_png))
        return True
    except Exception:
        return False

def draw_networkx_fallback(consensus: GeneralGraph, labels, title, out_png: pathlib.Path):
    """Fallback: draw directed as -> and bidirected as two arrows."""
    dir_edges, bi_edges, und_edges = [], [], []
    for e in consensus.get_graph_edges():
        s = e.get_node1().get_name()
        t = e.get_node2().get_name()
        ep1, ep2 = e.get_endpoint1().value, e.get_endpoint2().value
        if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
            dir_edges.append((s, t))
        elif ep1 == Endpoint.ARROW.value and ep2 == Endpoint.ARROW.value:
            bi_edges.append((s, t))  # we'll draw both directions
        elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
            und_edges.append((s, t))

    Gd = nx.DiGraph(); Gd.add_nodes_from(labels)
    Gd.add_edges_from(dir_edges)
    # bidirected: add both ways
    Gd.add_edges_from([(u, v) for (u, v) in bi_edges] + [(v, u) for (u, v) in bi_edges])

    pos = nx.spring_layout(Gd, seed=42)
    plt.figure(figsize=(11, 8))
    nx.draw_networkx_nodes(Gd, pos, node_color="#cfe8ff", node_size=900, edgecolors="#1b4965")
    nx.draw_networkx_labels(Gd, pos, font_size=9)

    if Gd.edges():
        nx.draw_networkx_edges(Gd, pos, edgelist=list(Gd.edges()),
                               arrows=True, arrowstyle="->", arrowsize=16, width=1.9)
    if und_edges:
        Gu = nx.Graph(); Gu.add_nodes_from(labels); Gu.add_edges_from(und_edges)
        nx.draw_networkx_edges(Gu, pos, edgelist=und_edges, style="dashed", width=1.5)

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

def run_discovery_once(df_boot: pd.DataFrame):
    """Call your Discovery wrapper and return its ADMG object."""
    disc = Discovery(lamda=LAMBDA)
    G = disc.discover_admg(
        data=df_boot,
        admg_class=ADMG_CLASS,
        w_threshold=W_THRESHOLD,
        num_restarts=NUM_RESTARTS,
        verbose=VERBOSE_BOOT
    )
    return G

def extract_admg_mats(G, labels):
    """
    Try to obtain directed and bidirected adjacencies from your ADMG object.
    Returns (D, BI) as {0,1} matrices aligned to 'labels'.
    We try several common shapes: edge lists, adjacency arrays, attributes.
    """
    p = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    D = np.zeros((p, p), dtype=int)   # i->j
    BI = np.zeros((p, p), dtype=int)  # i<->j (symmetric)

    def as_idx(u):
        if isinstance(u, int):
            return u
        return idx.get(str(u), None)

    def add_dir(u, v):
        i, j = as_idx(u), as_idx(v)
        if i is not None and j is not None and i != j:
            D[i, j] = 1

    def add_bi(u, v):
        i, j = as_idx(u), as_idx(v)
        if i is not None and j is not None and i != j:
            BI[i, j] = 1
            BI[j, i] = 1

    # --- edge-list patterns
    for attr in ("directed_edges", "dir_edges", "arcs"):
        if hasattr(G, attr):
            for (u, v) in getattr(G, attr) or []:
                add_dir(u, v)
    for attr in ("bidirected_edges", "bi_edges", "confounded"):
        if hasattr(G, attr):
            for (u, v) in getattr(G, attr) or []:
                add_bi(u, v)

    # --- adjacency patterns
    for attr in ("A_dir", "adj_dir", "directed", "D"):
        if hasattr(G, attr) and isinstance(getattr(G, attr), np.ndarray):
            A = (getattr(G, attr) != 0).astype(int)
            # assume A is ordered like 'labels'
            D |= (A > 0).astype(int)

    for attr in ("A_bi", "adj_bi", "bidirected", "U"):
        if hasattr(G, attr) and isinstance(getattr(G, attr), np.ndarray):
            A = (getattr(G, attr) != 0).astype(int)
            BI |= ((A + A.T) > 0).astype(int)

    # --- generic edges with a 'type' tag
    if (not D.any()) and (not BI.any()) and hasattr(G, "edges"):
        try:
            for e in G.edges:
                if isinstance(e, tuple) and len(e) >= 2:
                    u, v = e[:2]
                    if len(e) >= 3 and isinstance(e[2], str) and "bi" in e[2].lower():
                        add_bi(u, v)
                    else:
                        add_dir(u, v)
        except Exception:
            pass

    return D, BI

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

    # Load & clean
    df = pd.read_csv(args.data)
    df = clean_df(df)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns after cleaning.")
    labels = df.columns.tolist()
    n, p = df.shape
    print(f"[ADMG-BOOT] Data {df.shape} | n_boot={args.n_boot}")

    # -------- bootstrap counters --------
    B_eff = 0
    skel_counts = np.zeros((p, p), dtype=int)  # unordered pairs (a<b): any of ->, <-, or <-> present
    dir_counts  = np.zeros((p, p), dtype=int)  # ordered counts for i->j
    bi_counts   = np.zeros((p, p), dtype=int)  # unordered counts for i<->j, store in [a,b] with a<b

    rng = np.random.default_rng(RNG_SEED)

    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)
        df_b = df.iloc[rows].reset_index(drop=True)

        try:
            G = run_discovery_once(df_b)
        except Exception as e:
            print(f"[ADMG-BOOT] replicate {b}: FAILED -> {e}")
            continue

        D, BI = extract_admg_mats(G, labels)
        # skeleton pairs that appear in this bootstrap
        present_pairs = set()
        for i in range(p):
            for j in range(p):
                if i == j: continue
                if D[i, j] == 1 or D[j, i] == 1 or BI[i, j] == 1:
                    a, bb = (i, j) if i < j else (j, i)
                    present_pairs.add((a, bb))
        for a, bb in present_pairs:
            skel_counts[a, bb] += 1

        # directional counts
        for i in range(p):
            for j in range(p):
                if i == j: continue
                if D[i, j] == 1 and D[j, i] == 0:
                    dir_counts[i, j] += 1

        # bidirected counts (unordered)
        for i in range(p):
            for j in range(i + 1, p):
                if BI[i, j] == 1 or BI[j, i] == 1:
                    bi_counts[i, j] += 1

        B_eff += 1

    if B_eff == 0:
        raise RuntimeError("All ADMG bootstraps failed; no consensus computed.")

    # -------- probabilities --------
    skel_prob = skel_counts.astype(float) / B_eff

    # conditional probabilities given the pair exists
    same_dir_cond = np.full((p, p), np.nan, dtype=float)  # P(i->j | pair)
    bi_cond       = np.full((p, p), np.nan, dtype=float)  # P(i<->j | pair) (store in [a,b] with a<b)

    for a in range(p):
        for b in range(a + 1, p):
            denom = skel_counts[a, b]
            if denom > 0:
                # directed both ways
                same_dir_cond[a, b] = dir_counts[a, b] / denom
                same_dir_cond[b, a] = dir_counts[b, a] / denom
                # bidirected (unordered)
                bi_cond[a, b] = bi_counts[a, b] / denom

    # -------- consensus ADMG (GeneralGraph) --------
    nodes = [GraphNode(lbl) for lbl in labels]
    consensus = GeneralGraph(nodes)

    for a in range(p):
        for b in range(a + 1, p):
            sp = skel_prob[a, b]
            if sp < EDGE_THR:
                continue

            pab = 0.0 if np.isnan(same_dir_cond[a, b]) else same_dir_cond[a, b]  # a->b
            pba = 0.0 if np.isnan(same_dir_cond[b, a]) else same_dir_cond[b, a]  # b->a
            pbi = 0.0 if np.isnan(bi_cond[a, b])       else bi_cond[a, b]        # a<->b

            # choose the most probable type that exceeds its threshold
            if pbi >= pab and pbi >= pba and pbi >= BI_THR:
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.ARROW, Endpoint.ARROW))  # bidirected
            elif pab >= pba and pab >= DIR_THR:
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.ARROW))   # a->b
            elif pba > pab and pba >= DIR_THR:
                consensus.add_edge(Edge(nodes[b], nodes[a], Endpoint.TAIL, Endpoint.ARROW))   # b->a
            else:
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.TAIL))    # undirected

    # -------- save consensus PNG --------
    png_consensus = outdir / "admg_boot_consensus.png"
    if not try_graphviz_png(consensus, labels, png_consensus):
        # fallback draw
        draw_networkx_fallback(consensus, labels, "ADMG bootstrap consensus", png_consensus)
    print(f"[ADMG-BOOT] Saved: {png_consensus.name}")

    # -------- annotated PNG (skeleton%, same-type% | pair) --------
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
                # i->j
                sd = same_dir_cond[i, j]
                same_txt = "—" if np.isnan(sd) else f"{sd*100:.0f}%"
                text = f"({sk*100:.0f}%, {same_txt})"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], text)

            elif ep1 == Endpoint.ARROW.value and ep2 == Endpoint.ARROW.value:
                # bidirected
                pbi = bi_cond[a, b]
                bi_txt = "—" if np.isnan(pbi) else f"{pbi*100:.0f}%"
                text = f"({sk*100:.0f}%, {bi_txt})"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], text)

            elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
                # undirected
                text = f"({sk*100:.0f}%, —)"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], text)

        png_annot = outdir / "admg_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[ADMG-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[ADMG-BOOT] Annotation failed: {e}")

    # optional: debug matrices
    np.save(outdir / "admg_boot_skeleton_prob.npy", skel_prob)
    np.save(outdir / "admg_boot_dir_prob.npy", same_dir_cond)  # conditional dir probs
    np.save(outdir / "admg_boot_bi_prob.npy", bi_cond)         # conditional bi probs

    print(f"[ADMG-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[ADMG-BOOT] Outputs in: {outdir}")

if __name__ == "__main__":
    main()
