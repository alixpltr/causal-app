import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# causal-learn
from causallearn.search.ScoreBased.GES import ges
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.utils.GraphUtils import GraphUtils

# fallback drawing
import networkx as nx

# ---------- helpers ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric-only; drop NaN/Inf rows."""
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def map_name_to_label(nm: str, labels) -> str:
    """Map 'Xk' to labels[k-1] if possible; otherwise return nm."""
    if isinstance(nm, str) and nm.startswith("X"):
        try:
            idx = int(nm[1:]) - 1
            if 0 <= idx < len(labels):
                return labels[idx]
        except Exception:
            pass
    return nm

def try_graphviz_png(cg_or_graph, labels, out_png: pathlib.Path) -> bool:
    """Use Graphviz via pydot for clean layout. Return True if success."""
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

def draw_networkx(directed_edges, undirected_edges, labels, title, out_png: pathlib.Path):
    """Fallback: draw with NetworkX (arrows for directed, dashed for undirected)."""
    Gd = nx.DiGraph()
    Gd.add_nodes_from(labels)
    Gd.add_edges_from(directed_edges)
    pos = nx.spring_layout(Gd, seed=17)

    plt.figure(figsize=(11, 8))
    nx.draw_networkx_nodes(Gd, pos, node_color="#cfe8ff", node_size=900, edgecolors="#1b4965")
    nx.draw_networkx_labels(Gd, pos, font_size=9)

    if directed_edges:
        nx.draw_networkx_edges(Gd, pos, edgelist=directed_edges,
                               arrows=True, arrowstyle="->", arrowsize=16, width=1.9)
    if undirected_edges:
        Gu = nx.Graph()
        Gu.add_nodes_from(labels)
        Gu.add_edges_from(undirected_edges)
        nx.draw_networkx_edges(Gu, pos, edgelist=undirected_edges, style="dashed", width=1.5)

    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

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
        e.set("labelfontname", "Helvetica")
        e.set("labeldistance", "1.6")
        e.set("labelangle", "0")

def endpoints_to_orientation(ep_i, ep_j):
    """Return 'i2j', 'j2i', 'undirected', or 'adj' from endpoint types."""
    Ai, Aj = Endpoint.ARROW.value, Endpoint.TAIL.value
    if ep_i == Ai and ep_j == Aj: return "i2j"
    if ep_i == Aj and ep_j == Ai: return "j2i"
    if ep_i == Endpoint.TAIL.value and ep_j == Endpoint.TAIL.value: return "undirected"
    if ep_i == Endpoint.ARROW.value and ep_j == Endpoint.ARROW.value: return "adj"
    return "adj"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input CSV")
    ap.add_argument("--outdir", required=True, help="Directory to save outputs")
    ap.add_argument("--n_boot", type=int, default=20, help="Bootstrap replicates")
    ap.add_argument("--edge_thr", type=float, default=0.5, help="Include edge if skeleton prob ≥ this")
    ap.add_argument("--dir_thr", type=float, default=0.5, help="Orient edge if conditional dir prob ≥ this")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _ = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load & clean data
    df = pd.read_csv(args.data)
    df = clean_df(df)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns after cleaning.")
    X = df.to_numpy()
    labels = df.columns.tolist()
    n, p = X.shape

    label_to_idx = {c: i for i, c in enumerate(labels)}

    # ---------- bootstrap ----------
    B_eff = 0  # successful runs
    skel_counts = np.zeros((p, p), dtype=int)  # unordered (use [a,b] with a<b)
    dir_counts  = np.zeros((p, p), dtype=int)  # ordered (i->j)

    rng = np.random.default_rng(seed=12345)
    print(f"[GES-BOOT] Data {X.shape} | n_boot={args.n_boot}")

    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)
        Xb = X[rows]
        try:
            cg = ges(Xb)["G"]
        except Exception as e:
            print(f"[GES-BOOT] replicate {b}: FAILED -> {e}")
            continue

        B_eff += 1

        for edge in cg.get_graph_edges():
            n1 = map_name_to_label(edge.get_node1().get_name(), labels)
            n2 = map_name_to_label(edge.get_node2().get_name(), labels)
            if n1 not in label_to_idx or n2 not in label_to_idx: continue
            i, j = label_to_idx[n1], label_to_idx[n2]
            if i == j: continue

            ep1 = edge.get_endpoint1().value
            ep2 = edge.get_endpoint2().value
            orient = endpoints_to_orientation(ep1, ep2)

            # skeleton count (unordered)
            a, bpair = (i, j) if i < j else (j, i)
            skel_counts[a, bpair] += 1

            # direction count (ordered)
            if orient == "i2j":
                dir_counts[i, j] += 1
            elif orient == "j2i":
                dir_counts[j, i] += 1
            # 'undirected'/'adj' do not increment direction

    if B_eff == 0:
        raise RuntimeError("All GES bootstraps failed; no consensus computed.")

    # ---------- probabilities ----------
    # skeleton prob over B_eff (unordered)
    skel_prob = skel_counts.astype(float) / B_eff

    # conditional same-dir prob given pair appears:
    # P(i->j | pair exists) = dir_counts[i,j] / skel_counts[a,b]
    same_dir_cond = np.full((p, p), np.nan, dtype=float)
    for a in range(p):
        for b in range(a+1, p):
            denom = skel_counts[a, b]
            if denom > 0:
                same_dir_cond[a, b] = dir_counts[a, b] / denom     # a->b | pair
                same_dir_cond[b, a] = dir_counts[b, a] / denom     # b->a | pair

    # ---------- consensus graph ----------
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
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.TAIL))   # undirected

    # ---------- save consensus PNG ----------
    png_consensus = outdir / "ges_boot_consensus.png"
    if not try_graphviz_png(consensus, labels, png_consensus):
        # split lists for fallback draw
        dir_list, und_list = [], []
        for e in consensus.get_graph_edges():
            s, t = e.get_node1().get_name(), e.get_node2().get_name()
            ep1, ep2 = e.get_endpoint1().value, e.get_endpoint2().value
            if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
                dir_list.append((s, t))
            elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
                und_list.append((s, t))
        draw_networkx(dir_list, und_list, labels, "GES bootstrap consensus", png_consensus)
    print(f"[GES-BOOT] Saved: {png_consensus.name}")

    # ---------- annotated PNG (skeleton%, same-dir% | pair) ----------
    try:
        pyd = GraphUtils.to_pydot(consensus, labels=labels)
        label2id = pydot_label_map(pyd)

        for e in consensus.get_graph_edges():
            n1 = e.get_node1().get_name()
            n2 = e.get_node2().get_name()
            i, j = labels.index(n1), labels.index(n2)
            a, b = (i, j) if i < j else (j, i)

            sk = skel_prob[a, b]                 # in [0,1]
            ep1, ep2 = e.get_endpoint1().value, e.get_endpoint2().value

            if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
                sd = same_dir_cond[i, j]         # conditional same-dir prob
                same_txt = "-" if np.isnan(sd) else f"{int(round(sd * 100))}%"
                text = f"({int(round(sk * 100))}%, {same_txt})"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], text)
            elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
                pab = same_dir_cond[i, j]
                pba = same_dir_cond[j, i]

                if np.isnan(pab) and np.isnan(pba):
                    dir_txt = "-"
                else:
                    best_dir = max(pab if not np.isnan(pab) else 0.0,
                                pba if not np.isnan(pba) else 0.0)
                    dir_txt = f"{int(round(best_dir * 100))}%"

                text = f"({int(round(sk * 100))}%, {dir_txt})"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], text)


        png_annot = outdir / "ges_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[GES-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[GES-BOOT] Annotation failed: {e}")

    print(f"[GES-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[GES-BOOT] Outputs in: {outdir}")

if __name__ == "__main__":
    main()
