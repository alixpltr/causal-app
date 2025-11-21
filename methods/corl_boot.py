# methods/corl_boot.py
import os
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# prefer torch backend so CORL uses PyTorch (works fine on CPU)
os.environ.setdefault("CASTLE_BACKEND", "pytorch")

# gCastle
from castle.algorithms import CORL

# Graphviz via causallearn (nice layout + endpoint control)
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils

# fallback drawer
import networkx as nx


# ----------------- helpers -----------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric-only; drop NaN/Inf rows."""
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return df


def standardize_np(X: np.ndarray) -> np.ndarray:
    X = X.astype(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd


def try_graphviz_png(cg_or_graph, labels, out_png: pathlib.Path) -> bool:
    """Render a causallearn graph to PNG via Graphviz/pydot."""
    try:
        pyd = GraphUtils.to_pydot(
            cg_or_graph.G if hasattr(cg_or_graph, "G") else cg_or_graph,
            labels=labels
        )
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pyd.write_png(str(out_png))
        return True
    except Exception as e:
        print(f"[CORL-BOOT] Graphviz render failed: {e}")
        return False


def draw_networkx_fallback(dir_edges, und_edges, labels, title, out_png: pathlib.Path):
    """Fallback: draw with NetworkX (arrows for directed, dashed for undirected)."""
    Gd = nx.DiGraph()
    Gd.add_nodes_from(labels)
    Gd.add_edges_from(dir_edges)
    pos = nx.spring_layout(Gd, seed=42)

    plt.figure(figsize=(11, 8))
    nx.draw_networkx_nodes(Gd, pos, node_color="#E6F0FF", node_size=900, edgecolors="#345")
    nx.draw_networkx_labels(Gd, pos, font_size=9)

    if dir_edges:
        nx.draw_networkx_edges(Gd, pos, edgelist=dir_edges,
                               arrows=True, arrowstyle="->", arrowsize=16, width=1.8)

    if und_edges:
        Gu = nx.Graph()
        Gu.add_nodes_from(labels)
        Gu.add_edges_from(und_edges)
        nx.draw_networkx_edges(Gu, pos, edgelist=und_edges, style="dashed", width=1.5)

    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def pydot_label_map(pyd):
    """Map displayed node label -> internal node id (strings) for robust edge labeling."""
    m = {}
    for n in pyd.get_nodes():
        node_id = str(n.get_name()).strip('"').strip()
        label = n.get("label")
        label = node_id if label is None else str(label).strip('"').strip()
        m[label] = node_id
    return m


def set_edge_xlabel(pyd, src_id, dst_id, text):
    """Attach external label to an edge regardless of quoting order."""
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


# ----------------- CORL runner -----------------
def run_corl_adj(X: np.ndarray, iterations: int, batch_size: int, embed_dim: int,
                 encoder: str, decoder: str, thr: float | None) -> np.ndarray:
    """
    Fit CORL and return a binary adjacency matrix (i->j = 1 means edge i→j).
    If thr is not None, use (A_raw > thr); otherwise use (A_raw != 0).
    """
    n, p = X.shape
    corl = CORL(
        encoder_name=encoder,
        decoder_name=decoder,
        reward_mode="episodic",
        reward_regression_type="LR",
        batch_size=batch_size,
        input_dim=p,
        embed_dim=embed_dim,
        iteration=iterations,
        device_type="cpu",   # keep CPU for website portability
    )
    corl.learn(X)

    A_raw = np.asarray(corl.causal_matrix, dtype=float)
    if thr is None:
        A = (A_raw != 0).astype(int)
    else:
        A = (A_raw > thr).astype(int)
    np.fill_diagonal(A, 0)
    return A


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input CSV")
    ap.add_argument("--outdir", required=True, help="Directory to save outputs")
    ap.add_argument("--n_boot", type=int, default=20, help="Bootstrap replicates")
    ap.add_argument("--edge_thr", type=float, default=0.5, help="Include edge if skeleton prob ≥ this")
    ap.add_argument("--dir_thr", type=float, default=0.5, help="Orient edge if conditional dir prob ≥ this")

    # a few CORL knobs
    ap.add_argument("--iterations", type=int, default=1000, help="CORL iterations")
    ap.add_argument("--batch_size", type=int, default=64, help="CORL batch size")
    ap.add_argument("--embed_dim", type=int, default=64, help="CORL embedding dimension")
    ap.add_argument("--encoder", default="transformer", choices=["transformer", "mlp"], help="CORL encoder")
    ap.add_argument("--decoder", default="lstm", choices=["lstm", "mlp"], help="CORL decoder")
    ap.add_argument("--thr", type=float, default=0.0, help="Threshold applied to CORL causal_matrix (None => !=0)")
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
    X = standardize_np(df.to_numpy(dtype=float))
    n, p = X.shape

    print(f"[CORL-BOOT] Data {X.shape} | n_boot={args.n_boot} | "
          f"edge_thr={args.edge_thr}, dir_thr={args.dir_thr}")

    # --- bootstrap tallies ---
    B_eff = 0
    skel_counts = np.zeros((p, p), dtype=int)  # unordered (a<b): pair present (either direction)
    dir_counts  = np.zeros((p, p), dtype=int)  # ordered: i->j observed

    rng = np.random.default_rng(seed=12345)
    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)
        Xb = X[rows, :]
        try:
            Ab = run_corl_adj(
                Xb,
                iterations=args.iterations,
                batch_size=args.batch_size,
                embed_dim=args.embed_dim,
                encoder=args.encoder,
                decoder=args.decoder,
                thr=(None if args.thr is None else float(args.thr)),
            )
        except Exception as e:
            print(f"[CORL-BOOT] replicate {b}: FAILED -> {e}")
            continue

        # update counts
        for i in range(p):
            for j in range(i+1, p):
                # skeleton presence (either direction)
                if Ab[i, j] == 1 or Ab[j, i] == 1:
                    skel_counts[i, j] += 1
                # direction counts
                if Ab[i, j] == 1:
                    dir_counts[i, j] += 1
                if Ab[j, i] == 1:
                    dir_counts[j, i] += 1

        B_eff += 1

    if B_eff == 0:
        raise RuntimeError("All CORL bootstraps failed; no consensus computed.")

    # --- probabilities ---
    skel_prob = skel_counts.astype(float) / B_eff  # unordered
    same_dir_cond = np.full((p, p), np.nan, dtype=float)
    for a in range(p):
        for b in range(a+1, p):
            denom = skel_counts[a, b]
            if denom > 0:
                same_dir_cond[a, b] = dir_counts[a, b] / denom   # P(a->b | pair)
                same_dir_cond[b, a] = dir_counts[b, a] / denom   # P(b->a | pair)

    # --- consensus graph in causallearn GeneralGraph ---
    nodes = [GraphNode(lbl) for lbl in labels]
    consensus = GeneralGraph(nodes)

    for a in range(p):
        for b in range(a+1, p):
            sp = skel_prob[a, b]
            if sp < args.edge_thr:
                continue

            pab = 0.0 if np.isnan(same_dir_cond[a, b]) else same_dir_cond[a, b]
            pba = 0.0 if np.isnan(same_dir_cond[b, a]) else same_dir_cond[b, a]

            if pab >= pba and pab >= args.dir_thr:
                # a -> b
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.ARROW))
            elif pba > pab and pba >= args.dir_thr:
                # b -> a
                consensus.add_edge(Edge(nodes[b], nodes[a], Endpoint.TAIL, Endpoint.ARROW))
            else:
                # undirected if edge is frequent but direction is inconsistent
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.TAIL))

    # --- save arrays for the website ---
    np.save(outdir / "corl_boot_skeleton_prob.npy", skel_prob)
    np.save(outdir / "corl_boot_dir_prob.npy", same_dir_cond)

    # --- consensus PNG ---
    png_consensus = outdir / "corl_boot_consensus.png"
    if try_graphviz_png(consensus, labels, png_consensus):
        print(f"[CORL-BOOT] Saved: {png_consensus.name}")
    else:
        # split for fallback
        dir_list, und_list = [], []
        for e in consensus.get_graph_edges():
            s, t = e.get_node1().get_name(), e.get_node2().get_name()
            ep1, ep2 = e.get_endpoint1().value, e.get_endpoint2().value
            if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
                dir_list.append((s, t))
            elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
                und_list.append((s, t))
        draw_networkx_fallback(dir_list, und_list, labels, "CORL bootstrap consensus", png_consensus)
        print(f"[CORL-BOOT] Saved (fallback): {png_consensus.name}")

    # --- annotated PNG (skeleton%, dir% | pair) ---
    try:
        pyd = GraphUtils.to_pydot(consensus, labels=labels)
        label2id = pydot_label_map(pyd)

        for e in consensus.get_graph_edges():
            n1 = e.get_node1().get_name()
            n2 = e.get_node2().get_name()
            i = labels.index(n1)
            j = labels.index(n2)
            a, b = (i, j) if i < j else (j, i)

            sk = skel_prob[a, b]  # in [0,1]
            ep1, ep2 = e.get_endpoint1().value, e.get_endpoint2().value

            if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
                # i -> j
                sd = same_dir_cond[i, j]
                dir_txt = "—" if np.isnan(sd) else f"{sd*100:.0f}%"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, {dir_txt})")

            elif ep1 == Endpoint.ARROW.value and ep2 == Endpoint.TAIL.value:
                # j -> i (edge stored as n1–n2, but direction is j->i)
                sd = same_dir_cond[j, i]
                dir_txt = "—" if np.isnan(sd) else f"{sd*100:.0f}%"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, {dir_txt})")

            else:
                # undirected: show only skeleton %
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, —)")

        png_annot = outdir / "corl_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[CORL-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[CORL-BOOT] Annotation failed: {e}")

    print(f"[CORL-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[CORL-BOOT] Outputs in: {outdir}")


if __name__ == "__main__":
    main()
