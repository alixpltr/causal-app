# methods/shapley_pc_boot.py
import argparse
import pathlib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from datetime import datetime

# Shapley-PC (your local package)
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
SHAPLEY_DIR = HERE / "ShapleyPC"  # the folder that contains PC.py and spc.py

try:
    # prefer package-style import if your folder is a package
    from ShapleyPC.PC import pc
except ModuleNotFoundError:
    # fall back to adding the folder itself to sys.path so "import spc" in PC.py works
    if (SHAPLEY_DIR / "PC.py").exists():
        sys.path.insert(0, str(SHAPLEY_DIR))
        from ShapleyPC import pc
    else:
        raise

# causallearn graph utils
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils

# fallback drawing
import networkx as nx

# ---------- small helpers ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def standardize_np(X: np.ndarray) -> np.ndarray:
    X = X.astype(float)
    m = X.mean(axis=0)
    s = X.std(axis=0)
    s[s == 0.0] = 1.0
    return (X - m) / s

def map_name_to_label(nm: str, labels) -> str:
    """Map 'Xk' -> labels[k-1] if present; otherwise keep nm."""
    if isinstance(nm, str) and nm.startswith("X"):
        try:
            idx = int(nm[1:]) - 1
            if 0 <= idx < len(labels):
                return labels[idx]
        except Exception:
            pass
    return nm

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
                edges = [e]
                break
    for e in edges:
        e.set("xlabel", text)
        e.set("labelfontsize", "10")
        e.set("labelfontcolor", "black")
        e.set("labeldistance", "1.6")
        e.set("labelangle", "0")

def endpoints_to_orientation(ep_i, ep_j):
    """Return "i2j", "j2i", "undirected" or "adj" from Endpoint enums."""
    Ai, Aj = Endpoint.ARROW.value, Endpoint.TAIL.value
    if ep_i == Aj and ep_j == Ai: return "i2j"         # i tail, j arrow → i -> j
    if ep_i == Ai and ep_j == Aj: return "j2i"         # j -> i
    if ep_i == Endpoint.TAIL.value and ep_j == Endpoint.TAIL.value: return "undirected"
    if ep_i == Endpoint.ARROW.value and ep_j == Endpoint.ARROW.value: return "adj"
    return "adj"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input CSV")
    ap.add_argument("--outdir", required=True, help="Directory to save outputs")
    ap.add_argument("--n_boot", type=int, default=20, help="Bootstrap replicates")
    ap.add_argument("--alpha", type=float, default=0.01, help="PC/Shapley-PC alpha")
    ap.add_argument("--edge_thr", type=float, default=0.50, help="Keep edge if skeleton prob ≥ this")
    ap.add_argument("--dir_thr", type=float, default=0.50, help="Orient if conditional dir prob ≥ this")
    # You can expose more Shapley-PC knobs here if needed
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _ = datetime.now().strftime("%Y%m%d_%H%M%S")

    # load & clean
    df = pd.read_csv(args.data)
    df = clean_df(df)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns after cleaning.")
    labels = df.columns.tolist()
    X_full = standardize_np(df.to_numpy())
    n, p = X_full.shape

    print(f"[ShapleyPC-BOOT] Data {X_full.shape} | n_boot={args.n_boot} | alpha={args.alpha}")

    # --- bootstrap tallies (final graph, not intermediate) ---
    B_eff = 0  # successful runs
    skel_counts = np.zeros((p, p), dtype=int)   # unordered a<b : edge present
    dir_counts  = np.zeros((p, p), dtype=int)   # ordered i->j
    pattern_hist = defaultdict(Counter)         # unordered: (Endpoint_i, Endpoint_j) tallies

    rng = np.random.default_rng(seed=12345)

    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)
        Xb = X_full[rows]

        try:
            cg = pc(
                Xb,
                alpha=args.alpha,
                indep_test='fisherz',
                stable=True,
                uc_rule=3,
                uc_priority=2,
                selection='bot',
                show_progress=False,
                verbose=False
            )
            G = cg.G
        except Exception as e:
            print(f"[ShapleyPC-BOOT] replicate {b} FAILED -> {e}")
            continue

        B_eff += 1
        label_to_idx = {c: i for i, c in enumerate(labels)}

        for edge in G.get_graph_edges():
            n1 = map_name_to_label(edge.get_node1().get_name(), labels)
            n2 = map_name_to_label(edge.get_node2().get_name(), labels)
            if n1 not in label_to_idx or n2 not in label_to_idx:
                continue
            i, j = label_to_idx[n1], label_to_idx[n2]
            if i == j:
                continue

            ep1 = edge.get_endpoint1().value
            ep2 = edge.get_endpoint2().value
            orient = endpoints_to_orientation(ep1, ep2)

            a, bpair = (i, j) if i < j else (j, i)
            skel_counts[a, bpair] += 1

            # record detailed pattern as seen in stored order (i<j)
            # store the endpoints as seen (from node 'a' toward 'bpair')
            if a == i:
                pattern_hist[(a, bpair)][(ep1, ep2)] += 1
            else:
                # flip endpoints if (j,i) is the stored edge order
                pattern_hist[(a, bpair)][(ep2, ep1)] += 1

            if orient == "i2j":
                dir_counts[i, j] += 1
            elif orient == "j2i":
                dir_counts[j, i] += 1
            # undirected/adj: do not increment dir_counts

    if B_eff == 0:
        raise RuntimeError("All Shapley-PC bootstraps failed; no consensus computed.")

    # probabilities
    skel_prob = skel_counts.astype(float) / B_eff
    same_dir_cond = np.full((p, p), np.nan, dtype=float)  # P(i->j | pair)
    for a in range(p):
        for b in range(a+1, p):
            denom = skel_counts[a, b]
            if denom > 0:
                same_dir_cond[a, b] = dir_counts[a, b] / denom
                same_dir_cond[b, a] = dir_counts[b, a] / denom

    # consensus (GeneralGraph with label-named nodes)
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
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.ARROW))  # a->b
            elif pba > pab and pba >= args.dir_thr:
                consensus.add_edge(Edge(nodes[b], nodes[a], Endpoint.TAIL, Endpoint.ARROW))  # b->a
            else:
                # if direction is weak, keep it undirected (CPDAG style)
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.TAIL))

    # save consensus PNG
    png_consensus = outdir / "shapley_pc_boot_consensus.png"
    if not try_graphviz_png(consensus, labels, png_consensus):
        # networkx fallback
        dir_list, und_list = [], []
        for e in consensus.get_graph_edges():
            s, t = e.get_node1().get_name(), e.get_node2().get_name()
            ep1, ep2 = e.get_endpoint1().value, e.get_endpoint2().value
            if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
                dir_list.append((s, t))
            elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
                und_list.append((s, t))
        Gd = nx.DiGraph(); Gd.add_nodes_from(labels); Gd.add_edges_from(dir_list)
        pos = nx.spring_layout(Gd, seed=42)
        plt.figure(figsize=(11,8))
        nx.draw_networkx_nodes(Gd, pos, node_color="#E6F0FF", node_size=900, edgecolors="#345")
        nx.draw_networkx_labels(Gd, pos, font_size=9)
        if dir_list:
            nx.draw_networkx_edges(Gd, pos, edgelist=dir_list, arrows=True, arrowstyle="->", arrowsize=16, width=1.9)
        if und_list:
            Gu = nx.Graph(); Gu.add_nodes_from(labels); Gu.add_edges_from(und_list)
            nx.draw_networkx_edges(Gu, pos, edgelist=und_list, style="dashed", width=1.5)
        plt.title("Shapley-PC bootstrap consensus (fallback)")
        plt.tight_layout()
        plt.savefig(png_consensus, dpi=150); plt.close()
    print(f"[ShapleyPC-BOOT] Saved: {png_consensus.name}")

    # annotated PNG (skeleton%, same-dir% | pair)
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
                sd = same_dir_cond[i, j]
                same_txt = "—" if np.isnan(sd) else f"{sd*100:.0f}%"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, {same_txt})")
            elif ep1 == Endpoint.ARROW.value and ep2 == Endpoint.TAIL.value:
                sd = same_dir_cond[j, i]
                same_txt = "—" if np.isnan(sd) else f"{sd*100:.0f}%"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, {same_txt})")
            else:
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, —)")

        png_annot = outdir / "shapley_pc_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[ShapleyPC-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[ShapleyPC-BOOT] Annotation failed: {e}")

    # save arrays
    np.save(outdir / "shapley_pc_boot_skeleton_prob.npy", skel_prob)
    np.save(outdir / "shapley_pc_boot_dir_prob.npy", same_dir_cond)

    print(f"[ShapleyPC-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[ShapleyPC-BOOT] Outputs in: {outdir}")

if __name__ == "__main__":
    main()
