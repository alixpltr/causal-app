# methods/fges_boot.py
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx  # fallback draw

# ---- your FGES implementation (must exist in methods/fges_impl.py) ----
from fges_impl import FGES

# ====== FIXED INTERNAL SETTINGS ======
EDGE_THR  = 0.5     # include edge if skeleton prob ≥ 0.5
DIR_THR   = 0.5     # orient edge if conditional dir prob ≥ 0.5
PENALTY   = 2.0     # Gaussian BIC penalty (like lambda)
RNG_SEED  = 12345

# ---------- helpers ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric-only; drop NaN/Inf rows."""
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def standardize(X: np.ndarray) -> np.ndarray:
    X = X.astype(float)
    X -= X.mean(axis=0)
    s = X.std(axis=0)
    s[s == 0.0] = 1.0
    return X / s

def try_graphviz_png(cg_or_graph, labels, out_png: pathlib.Path) -> bool:
    """Use Graphviz via pydot for clean layout. Return True if success."""
    try:
        # Build a causal-learn GeneralGraph from our consensus nx graph
        from causallearn.graph.GraphNode import GraphNode
        from causallearn.graph.GeneralGraph import GeneralGraph
        from causallearn.graph.Edge import Edge
        from causallearn.graph.Endpoint import Endpoint
        from causallearn.utils.GraphUtils import GraphUtils

        if isinstance(cg_or_graph, nx.DiGraph):
            nodes = [GraphNode(lbl) for lbl in labels]
            idx = {lbl: i for i, lbl in enumerate(labels)}
            GG = GeneralGraph(nodes)
            # directed edges
            for u, v in cg_or_graph.edges():
                GG.add_edge(Edge(nodes[idx[u]], nodes[idx[v]], Endpoint.TAIL, Endpoint.ARROW))
            # undirected edges carried via graph attribute 'undirected_pairs'
            for a, b in cg_or_graph.graph.get("undirected_pairs", []):
                GG.add_edge(Edge(nodes[idx[a]], nodes[idx[b]], Endpoint.TAIL, Endpoint.TAIL))
            target = GG
        else:
            target = cg_or_graph

        from causallearn.utils.GraphUtils import GraphUtils
        pyd = GraphUtils.to_pydot(
            target.G if hasattr(target, "G") else target,
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
    pos = nx.spring_layout(Gd, seed=42)

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

# -------- Gaussian BIC score (minimal) --------
class GaussianBICScore:
    """
    Minimal score object expected by your FGES implementation, with:
      - local_score_diff(y, x)
      - local_score_diff_parents(x, y, parent_indices)
    """
    def __init__(self, X: np.ndarray, penalty: float = PENALTY):
        self.X = X.astype(float)
        self.n, self.d = self.X.shape
        self.penalty = penalty

    def _bic_local(self, y: int, parents: list[int]) -> float:
        import math
        n = self.n
        yv = self.X[:, y]
        if len(parents) == 0:
            sigma2 = float(np.mean(yv ** 2))
            return -n * math.log(sigma2 + 1e-12) - self.penalty * 1 * math.log(n)
        A = self.X[:, parents]
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        resid = yv - A @ beta
        sigma2 = float(np.mean(resid ** 2))
        k = len(parents) + 1
        return -n * math.log(sigma2 + 1e-12) - self.penalty * k * math.log(n)

    def local_score_diff(self, y: int, x: int) -> float:
        return self._bic_local(y, [x]) - self._bic_local(y, [])

    def local_score_diff_parents(self, x: int, y: int, parent_indices: list[int]) -> float:
        Pa = list(parent_indices)
        if x in Pa:
            return 0.0
        return self._bic_local(y, Pa + [x]) - self._bic_local(y, Pa)

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

    # Load & clean
    df = pd.read_csv(args.data)
    df = clean_df(df)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns after cleaning.")
    X = df.to_numpy()
    labels = df.columns.tolist()
    n, p = X.shape
    label_to_idx = {c: i for i, c in enumerate(labels)}

    # ----- bootstrap counters -----
    B_eff = 0
    skel_counts = np.zeros((p, p), dtype=int)  # unordered pairs (a<b)
    dir_counts  = np.zeros((p, p), dtype=int)  # ordered arcs (i->j)

    rng = np.random.default_rng(seed=RNG_SEED)
    print(f"[FGES-BOOT] Data {X.shape} | n_boot={args.n_boot} | penalty={PENALTY}")

    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)
        Xb = standardize(X[rows])

        try:
            score = GaussianBICScore(Xb, penalty=PENALTY)
            variables = list(range(p))
            fges = FGES(variables=variables, score=score, knowledge=None, verbose=False)
            result = fges.search()
            G = result["graph"]  # expected nx.DiGraph with int nodes
        except Exception as e:
            print(f"[FGES-BOOT] replicate {b}: FAILED -> {e}")
            continue

        # Relabel nodes to column names
        mapping = {i: labels[i] for i in range(p)}
        G_lab = nx.relabel_nodes(G, mapping, copy=True)

        # Collect skeleton once per pair in this bootstrap
        present_pairs = set()
        for u, v in G_lab.edges():
            if u not in label_to_idx or v not in label_to_idx: 
                continue
            i, j = label_to_idx[u], label_to_idx[v]
            if i == j: 
                continue
            a, bb = (i, j) if i < j else (j, i)
            present_pairs.add((a, bb))
            dir_counts[i, j] += 1  # directional tally

        # increment skeleton counts once per pair
        for a, bb in present_pairs:
            skel_counts[a, bb] += 1

        B_eff += 1

    if B_eff == 0:
        raise RuntimeError("All FGES bootstraps failed; no consensus computed.")

    # ----- probabilities -----
    skel_prob = skel_counts.astype(float) / B_eff
    same_dir_cond = np.full((p, p), np.nan, dtype=float)
    for a in range(p):
        for b in range(a+1, p):
            denom = skel_counts[a, b]
            if denom > 0:
                same_dir_cond[a, b] = dir_counts[a, b] / denom     # a->b | pair
                same_dir_cond[b, a] = dir_counts[b, a] / denom     # b->a | pair

    # ----- consensus graph as nx.DiGraph + carry undirected pairs in metadata -----
    G_cons = nx.DiGraph()
    G_cons.add_nodes_from(labels)
    undirected_pairs = []

    for a in range(p):
        for b in range(a+1, p):
            sp = skel_prob[a, b]
            if sp < EDGE_THR:
                continue
            pab = 0.0 if np.isnan(same_dir_cond[a, b]) else same_dir_cond[a, b]
            pba = 0.0 if np.isnan(same_dir_cond[b, a]) else same_dir_cond[b, a]

            if pab >= pba and pab >= DIR_THR:
                G_cons.add_edge(labels[a], labels[b])  # a->b
            elif pba > pab and pba >= DIR_THR:
                G_cons.add_edge(labels[b], labels[a])  # b->a
            else:
                undirected_pairs.append((labels[a], labels[b]))

    G_cons.graph["undirected_pairs"] = undirected_pairs  # for Graphviz helper

    # ----- save consensus PNG -----
    png_consensus = outdir / "fges_boot_consensus.png"
    if not try_graphviz_png(G_cons, labels, png_consensus):
        dir_list = list(G_cons.edges())
        und_list = undirected_pairs
        draw_networkx(dir_list, und_list, labels, "FGES bootstrap consensus", png_consensus)
    print(f"[FGES-BOOT] Saved: {png_consensus.name}")

    # ----- annotated PNG (skeleton%, same-dir% | pair) -----
    try:
        # Build a causal-learn GeneralGraph from consensus for annotation
        from causallearn.graph.GraphNode import GraphNode
        from causallearn.graph.GeneralGraph import GeneralGraph
        from causallearn.graph.Edge import Edge
        from causallearn.graph.Endpoint import Endpoint
        from causallearn.utils.GraphUtils import GraphUtils

        nodes = [GraphNode(lbl) for lbl in labels]
        idx = {lbl: i for i, lbl in enumerate(labels)}
        GG = GeneralGraph(nodes)
        for u, v in G_cons.edges():
            GG.add_edge(Edge(nodes[idx[u]], nodes[idx[v]], Endpoint.TAIL, Endpoint.ARROW))
        for u, v in undirected_pairs:
            GG.add_edge(Edge(nodes[idx[u]], nodes[idx[v]], Endpoint.TAIL, Endpoint.TAIL))

        pyd = GraphUtils.to_pydot(GG, labels=labels)
        label2id = pydot_label_map(pyd)

        # attach labels
        for e in GG.get_graph_edges():
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

        png_annot = outdir / "fges_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[FGES-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[FGES-BOOT] Annotation failed: {e}")

    print(f"[FGES-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[FGES-BOOT] Outputs in: {outdir}")

if __name__ == "__main__":
    main()
