# methods/fci_boot.py
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime

# causal-learn
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge

# fallback draw
import networkx as nx

# ===== fixed internal settings (not exposed) =====
ALPHA    = 0.05   # FCI Fisher-Z
EDGE_THR = 0.50   # include pair in consensus if skeleton prob ≥ this
DIR_THR  = 0.50   # direct a->b if P(a->b | pair) ≥ this AND ≥ opposite
RNG_SEED = 12345

# ---------- helpers ----------
def clean_df_no_missing(df: pd.DataFrame) -> pd.DataFrame:
    """FCI (FisherZ) expects complete rows; keep numeric, drop NaNs/±inf."""
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return df

def map_name_to_label(nm: str, labels) -> str:
    """Map 'Xk' → labels[k-1] if possible; otherwise return nm."""
    if isinstance(nm, str) and nm.startswith("X"):
        try:
            idx = int(nm[1:]) - 1
            if 0 <= idx < len(labels):
                return labels[idx]
        except Exception:
            pass
    return nm

def try_graphviz_png(GG: GeneralGraph, labels, out_png: pathlib.Path) -> bool:
    try:
        pyd = GraphUtils.to_pydot(GG, labels=labels)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pyd.write_png(str(out_png))
        return True
    except Exception:
        return False

def draw_networkx_fallback(consensus: GeneralGraph, labels, title, out_png: pathlib.Path):
    """Fallback (loses circles/↔): draws only directed (tail→arrow) & undirected (tail–tail)."""
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

# normalize endpoints to pair order (a<b)
def oriented_endpoints_for_pair(i, j, ep1, ep2, a, b):
    """Return endpoints as they apply to the ordered pair (a->b)."""
    if i == a and j == b:
        return (ep1, ep2)
    else:
        return (ep2, ep1)

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

    # load & clean
    df = pd.read_csv(args.data)
    df = clean_df_no_missing(df)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns after cleaning.")
    labels = df.columns.tolist()
    X_full = df.to_numpy(dtype=float)
    n, p = X_full.shape
    label_to_idx = {c: i for i, c in enumerate(labels)}

    print(f"[FCI-BOOT] Data {X_full.shape} | n_boot={args.n_boot} | alpha={ALPHA}")

    # ---------- bootstrap counters ----------
    B_eff = 0
    # unordered skeleton counts (a<b)
    skel_counts = np.zeros((p, p), dtype=int)
    # ordered directed counts (i->j as tail→arrow)
    dir_counts  = np.zeros((p, p), dtype=int)
    # unordered endpoint distribution for deciding ambiguous consensus endpoints
    # counts_by_pair[(a,b)][(ep_left, ep_right)] -> int
    counts_by_pair = defaultdict(Counter)

    rng = np.random.default_rng(RNG_SEED)

    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)
        Xb = X_full[rows, :]

        try:
            G, _ = fci(Xb, fisherz, alpha=ALPHA, verbose=False)  # PAG
        except Exception as e:
            print(f"[FCI-BOOT] replicate {b}: FAILED -> {e}")
            continue

        # collect unique pairs present in this run
        present_pairs = set()

        for e in G.get_graph_edges():
            n1 = map_name_to_label(e.get_node1().get_name(), labels)
            n2 = map_name_to_label(e.get_node2().get_name(), labels)
            if n1 not in label_to_idx or n2 not in label_to_idx: 
                continue
            i = label_to_idx[n1]
            j = label_to_idx[n2]
            if i == j: 
                continue

            ep1 = e.get_endpoint1().value
            ep2 = e.get_endpoint2().value

            a, bpair = (i, j) if i < j else (j, i)
            present_pairs.add((a, bpair))

            # normalize endpoints to (a->b) orientation
            epL, epR = oriented_endpoints_for_pair(i, j, ep1, ep2, a, bpair)
            counts_by_pair[(a, bpair)][(epL, epR)] += 1

            # directional (tail→arrow) counts for conditional direction probs
            if epL == Endpoint.TAIL.value and epR == Endpoint.ARROW.value:
                dir_counts[a, bpair] += 1
            elif epL == Endpoint.ARROW.value and epR == Endpoint.TAIL.value:
                dir_counts[bpair, a] += 1

        # bump skeleton counts once per present pair
        for (a, bpair) in present_pairs:
            skel_counts[a, bpair] += 1

        B_eff += 1

    if B_eff == 0:
        raise RuntimeError("All FCI bootstraps failed; no consensus computed.")

    # ---------- probabilities ----------
    skel_prob = skel_counts.astype(float) / B_eff

    # conditional same-dir prob given pair exists:
    # P(a->b | pair) = dir_counts[a,b] / skel_counts[a,b]
    same_dir_cond = np.full((p, p), np.nan, dtype=float)
    for a in range(p):
        for b in range(a + 1, p):
            denom = skel_counts[a, b]
            if denom > 0:
                same_dir_cond[a, b] = dir_counts[a, b] / denom
                same_dir_cond[b, a] = dir_counts[b, a] / denom

    # ---------- consensus PAG-like graph ----------
    nodes = [GraphNode(lbl) for lbl in labels]
    consensus = GeneralGraph(nodes)

    for a in range(p):
        for b in range(a + 1, p):
            sp = skel_prob[a, b]
            if sp < EDGE_THR:
                continue

            pab = 0.0 if np.isnan(same_dir_cond[a, b]) else same_dir_cond[a, b]
            pba = 0.0 if np.isnan(same_dir_cond[b, a]) else same_dir_cond[b, a]

            # 1) Confident direction
            if pab >= pba and pab >= DIR_THR:
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.ARROW))  # a->b
                continue
            if pba > pab and pba >= DIR_THR:
                consensus.add_edge(Edge(nodes[b], nodes[a], Endpoint.TAIL, Endpoint.ARROW))  # b->a
                continue

            # 2) Not confident: pick MOST FREQUENT ambiguous endpoint combo
            #    Prefer among: TT (undirected), AA (bidirected), any with circles.
            ep_counts = counts_by_pair.get((a, b), Counter())
            if not ep_counts:
                # fallback: undirected
                consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.TAIL))
                continue

            # filter to "ambiguous" categories (exclude clear directions TA and AT)
            def is_ambiguous(pair):
                L, R = pair
                if (L == Endpoint.TAIL.value and R == Endpoint.ARROW.value):  # TA (a->b)
                    return False
                if (L == Endpoint.ARROW.value and R == Endpoint.TAIL.value):  # AT (b->a)
                    return False
                return True

            amb_only = {k: v for k, v in ep_counts.items() if is_ambiguous(k)}

            # if none ambiguous observed, show undirected as conservative default
            chosen = max(amb_only.items(), key=lambda kv: kv[1])[0] if amb_only else (Endpoint.TAIL.value, Endpoint.TAIL.value)

            epL, epR = chosen
            consensus.add_edge(Edge(nodes[a], nodes[b], Endpoint(epL), Endpoint(epR)))

    # ---------- save consensus PNG ----------
    png_consensus = outdir / "fci_boot_consensus.png"
    if not try_graphviz_png(consensus, labels, png_consensus):
        draw_networkx_fallback(consensus, labels, "FCI bootstrap consensus", png_consensus)
    print(f"[FCI-BOOT] Saved: {png_consensus.name}")

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
                sd = same_dir_cond[i, j]  # P(i->j | pair)
                same_txt = "—" if np.isnan(sd) else f"{sd*100:.0f}%"
                text = f"({sk*100:.0f}%, {same_txt})"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], text)
            else:
                # ambiguous / undirected / bidirected => show only skeleton%
                text = f"({sk*100:.0f}%, —)"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], text)

        png_annot = outdir / "fci_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[FCI-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[FCI-BOOT] Annotation failed: {e}")

    # optional: save raw arrays
    np.save(outdir / "fci_boot_skeleton_prob.npy", skel_prob)
    np.save(outdir / "fci_boot_dir_prob.npy", same_dir_cond)

    print(f"[FCI-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[FCI-BOOT] Outputs in: {outdir}")

if __name__ == "__main__":
    main()
