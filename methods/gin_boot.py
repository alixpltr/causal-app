# methods/gin_boot.py
import argparse
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict

from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils

# ---------------- helpers ----------------

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.select_dtypes(include=[np.number])
          .replace([np.inf, -np.inf], np.nan)
          .dropna(axis=0)
    )

def map_name_to_observed(nm: str, labels) -> str | None:
    """GIN nodes are typically 'X1'..'Xd' for observed; latent won't map."""
    if isinstance(nm, str) and nm.startswith("X"):
        try:
            idx = int(nm[1:]) - 1
            if 0 <= idx < len(labels):
                return labels[idx]
        except Exception:
            pass
    return None  # latent or unknown → ignore for consensus

def endpoints_to_orientation(ep_i, ep_j):
    Ai, Aj = Endpoint.ARROW.value, Endpoint.TAIL.value
    if ep_i == Aj and ep_j == Ai: return "i2j"    # i tail, j arrow → i -> j
    if ep_i == Ai and ep_j == Aj: return "j2i"    # j tail, i arrow → j -> i
    if ep_i == Endpoint.TAIL.value and ep_j == Endpoint.TAIL.value: return "undirected"
    return "amb"  # circles/arrows on both ends etc.

def pydot_label_map(pyd):
    m = {}
    for n in pyd.get_nodes():
        node_id = str(n.get_name()).strip('"').strip()
        lbl = n.get("label")
        lbl = node_id if lbl is None else str(lbl).strip('"').strip()
        m[lbl] = node_id
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

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input CSV")
    ap.add_argument("--outdir", required=True, help="Directory to save outputs")
    ap.add_argument("--n_boot", type=int, default=20, help="Bootstrap replicates")
    ap.add_argument("--edge_thr", type=float, default=0.50, help="Keep edge if skeleton prob ≥ this")
    ap.add_argument("--dir_thr", type=float, default=0.50, help="Orient if dir prob ≥ this and ≥ opposite")
    ap.add_argument("--sample_frac", type=float, default=1.0, help="Rows per bootstrap (0<frac≤1), sampled w/ replacement")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # load & clean
    df = pd.read_csv(args.data)
    df = clean_numeric(df)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns after cleaning.")
    labels = df.columns.tolist()
    X = df.to_numpy(dtype=float)
    n, p = X.shape

    print(f"[GIN-BOOT] Data {X.shape} | n_boot={args.n_boot}")

    # tallies (observed-observed only)
    B_eff = 0
    skel_counts = np.zeros((p, p), dtype=int)  # unordered: pair present
    dir_counts  = np.zeros((p, p), dtype=int)  # ordered: i->j
    pattern_hist = defaultdict(Counter)        # optional debug: store raw endpoint patterns

    rng = np.random.default_rng(args.seed)
    m = max(2, int(np.ceil(args.sample_frac * n)))
    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=m, endpoint=False)  # bootstrap rows (with replacement)
        Xb = X[rows, :]

        try:
            G_graph, _ = GIN(Xb)
        except Exception as e:
            print(f"[GIN-BOOT] replicate {b}: FAILED -> {e}")
            continue

        present_pairs = set()

        for e in G_graph.get_graph_edges():
            n1 = e.get_node1().get_name()
            n2 = e.get_node2().get_name()
            obs1 = map_name_to_observed(n1, labels)
            obs2 = map_name_to_observed(n2, labels)
            if obs1 is None or obs2 is None or obs1 == obs2:
                continue  # ignore latent edges and self loops

            i, j = labels.index(obs1), labels.index(obs2)
            a, bpair = (i, j) if i < j else (j, i)
            present_pairs.add((a, bpair))

            ep1 = e.get_endpoint1().value
            ep2 = e.get_endpoint2().value
            orient = endpoints_to_orientation(ep1, ep2)

            # store raw pattern (optional)
            pattern_hist[(a, bpair)][orient] += 1

            if orient == "i2j":
                dir_counts[i, j] += 1
            elif orient == "j2i":
                dir_counts[j, i] += 1
            # undirected/ambiguous: do not increment direction

        for (a, bpair) in present_pairs:
            skel_counts[a, bpair] += 1

        B_eff += 1

    if B_eff == 0:
        raise RuntimeError("All GIN bootstraps failed; no consensus computed.")

    # probabilities
    skel_prob = skel_counts.astype(float) / B_eff
    same_dir_cond = np.full((p, p), np.nan, dtype=float)
    for a in range(p):
        for b in range(a+1, p):
            denom = skel_counts[a, b]
            if denom > 0:
                same_dir_cond[a, b] = dir_counts[a, b] / denom
                same_dir_cond[b, a] = dir_counts[b, a] / denom

    # consensus graph on observed nodes
    nodes = [GraphNode(lbl) for lbl in labels]
    GG = GeneralGraph(nodes)
    for a in range(p):
        for b in range(a+1, p):
            sp = skel_prob[a, b]
            if sp < args.edge_thr:
                continue

            pab = 0.0 if np.isnan(same_dir_cond[a, b]) else same_dir_cond[a, b]
            pba = 0.0 if np.isnan(same_dir_cond[b, a]) else same_dir_cond[b, a]

            if pab >= pba and pab >= args.dir_thr:
                GG.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.ARROW))  # a->b
            elif pba > pab and pba >= args.dir_thr:
                GG.add_edge(Edge(nodes[b], nodes[a], Endpoint.TAIL, Endpoint.ARROW))  # b->a
            else:
                GG.add_edge(Edge(nodes[a], nodes[b], Endpoint.TAIL, Endpoint.TAIL))   # undirected

    # save consensus PNG
    try:
        pyd = GraphUtils.to_pydot(GG, labels=labels)
        png_cons = outdir / f"gin_boot_consensus_{ts}.png"
        pyd.write_png(str(png_cons))
        print(f"[GIN-BOOT] Saved: {png_cons.name}")
    except Exception as e:
        print(f"[GIN-BOOT] Consensus render failed: {e}")

    # save annotated PNG (skeleton%, dir%)
    try:
        pyd = GraphUtils.to_pydot(GG, labels=labels)
        label2id = pydot_label_map(pyd)

        for e in GG.get_graph_edges():
            n1 = e.get_node1().get_name()
            n2 = e.get_node2().get_name()
            i, j = labels.index(n1), labels.index(n2)
            a, b = (i, j) if i < j else (j, i)

            sk = skel_prob[a, b]  # in [0,1]
            ep1 = e.get_endpoint1().value
            ep2 = e.get_endpoint2().value

            if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
                sd = same_dir_cond[i, j]
                txt = f"({sk*100:.0f}%, {'—' if np.isnan(sd) else f'{sd*100:.0f}%'})"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], txt)
            elif ep1 == Endpoint.TAIL.value and ep2 == Endpoint.TAIL.value:
                txt = f"({sk*100:.0f}%, —)"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], txt)

        png_annot = outdir / f"gin_boot_consensus_annotated_{ts}.png"
        pyd.write_png(str(png_annot))
        print(f"[GIN-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[GIN-BOOT] Annotation failed: {e}")

    # optional: save raw arrays
    np.save(outdir / f"gin_boot_skeleton_prob_{ts}.npy", skel_prob)
    np.save(outdir  / f"gin_boot_dir_prob_{ts}.npy", same_dir_cond)
    print(f"[GIN-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[GIN-BOOT] Outputs in: {outdir}")

if __name__ == "__main__":
    main()
