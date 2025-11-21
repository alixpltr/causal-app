# methods/daggnn_boot.py
import argparse
import pathlib
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Graph rendering via causal-learn
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils

import networkx as nx  # fallback draw
# --- NetworkX 3.x compatibility shim (do this near the top) ---
import networkx as nx
import networkx.convert_matrix as nxc

# add nx.from_numpy_matrix if missing
if not hasattr(nx, "from_numpy_matrix") and hasattr(nx, "from_numpy_array"):
    nx.from_numpy_matrix = nx.from_numpy_array

# add convert_matrix.from_numpy_matrix if missing (repo uses this import)
if not hasattr(nxc, "from_numpy_matrix") and hasattr(nxc, "from_numpy_array"):
    nxc.from_numpy_matrix = nxc.from_numpy_array
# --- end shim ---


# ====== FIXED INTERNAL SETTINGS ======
WEIGHT_THR = 0.50   # binarize each run: edge i->j present if |A[i,j]| >= 0.30
EDGE_THR   = 0.50   # include in consensus if skeleton prob ≥ 0.5
DIR_THR    = 0.50   # orient in consensus if conditional dir prob ≥ 0.5
RNG_SEED   = 12345

# ---- DAG-GNN repo config tweaks (for speed & CPU safety) ----
def configure_daggnn():
    from DAG_from_GNN.config import CONFIG
    # Force CPU (safer in varied environments)
    CONFIG.no_cuda = True
    # Make it reasonably quick; tune if you like
    CONFIG.k_max_iter = 2     # fewer alternating steps
    CONFIG.epochs = 150
    CONFIG.h_tol = 1e-1
    CONFIG.batch_size = 50
    CONFIG.lr = 1e-3
    CONFIG.tau_A = 0.1
    CONFIG.c_A = 10
    CONFIG.encoder = "mlp"
    CONFIG.decoder = "mlp"
    # Graph threshold inside repo isn't used for our counting; we threshold ourselves.
    CONFIG.graph_threshold = WEIGHT_THR
    CONFIG.seed = 42
    return CONFIG

def patch_repo_main_numpy_to_cpu():
    """Ensure any .numpy() calls in the repo use .cpu().numpy() so CPU/CPU↔GPU is safe."""
    import pathlib
    main_path = pathlib.Path(__file__).parent / "DAG_from_GNN" / "__main__.py"
    if not main_path.exists():
        return
    txt = main_path.read_text(encoding="utf-8")
    new = txt.replace(".data.clone().numpy()", ".data.clone().cpu().numpy()")
    new = new.replace(".numpy()", ".cpu().numpy()")
    if new != txt:
        main_path.write_text(new, encoding="utf-8")

def run_repo_on_csv(csv_in_datasets: str) -> np.ndarray:
    """
    Run DAG-GNN repo training on a CSV saved under ./datasets and return the weighted adjacency A (np.ndarray).
    """
    CONFIG = configure_daggnn()
    CONFIG.data_filename = pathlib.Path(csv_in_datasets).name
    patch_repo_main_numpy_to_cpu()
    mod = importlib.reload(importlib.import_module("DAG_from_GNN.__main__"))
    if not hasattr(mod, "origin_A") or mod.origin_A is None:
        raise RuntimeError("DAG-GNN did not produce origin_A.")
    return mod.origin_A.data.clone().cpu().numpy()

# ---------- helpers ----------
def clean_df_no_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric columns, convert ±inf→NaN, then drop rows with NaN (DAG-GNN expects no NaNs)."""
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

    script_dir = pathlib.Path(__file__).parent
    datasets_dir = script_dir / "datasets"
    results_dir = script_dir / "results"
    datasets_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load & clean (no missing for DAG-GNN)
    df = pd.read_csv(args.data)
    df = clean_df_no_missing(df)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns after cleaning.")
    labels = df.columns.tolist()
    X = df.to_numpy()
    n, p = X.shape
    print(f"[DAGGNN-BOOT] Data {X.shape} | n_boot={args.n_boot} | thr={WEIGHT_THR}")

    # ---------- bootstrap counters ----------
    B_eff = 0
    skel_counts = np.zeros((p, p), dtype=int)  # unordered pairs (a<b)
    dir_counts  = np.zeros((p, p), dtype=int)  # ordered arcs (i->j)

    rng = np.random.default_rng(RNG_SEED)

    # Optional: run once on full data (not needed for consensus, but useful to sanity-check)
    # Save to datasets/ and run repo
    full_csv = datasets_dir / f"daggnn_full_{_ts}.csv"
    df.to_csv(full_csv, index=False)
    try:
        A_full = run_repo_on_csv(str(full_csv))  # weighted adjacency
        A_full_bin = (np.abs(A_full) >= WEIGHT_THR).astype(int)
        np.save(outdir / f"daggnn_full_weighted_{_ts}.npy", A_full)
        np.save(outdir / f"daggnn_full_binary_thr_{WEIGHT_THR:.2f}_{_ts}.npy", A_full_bin)
    except Exception as e:
        print(f"[DAGGNN-BOOT] Full run failed (continuing to bootstraps): {e}")

    # ---------- bootstraps ----------
    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)  # standard bootstrap, same size with replacement
        df_b = df.iloc[rows].reset_index(drop=True)
        boot_csv = datasets_dir / f"daggnn_boot_{b}_{_ts}.csv"
        df_b.to_csv(boot_csv, index=False)

        try:
            A_b = run_repo_on_csv(str(boot_csv))
        except Exception as e:
            print(f"[DAGGNN-BOOT] replicate {b}: FAILED -> {e}")
            continue

        # binarize this run
        A = (np.abs(A_b) >= WEIGHT_THR).astype(int)
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

        # direction only if exactly one direction present
        for i in range(p):
            for j in range(p):
                if i == j: continue
                if A[i, j] == 1 and A[j, i] == 0:
                    dir_counts[i, j] += 1

        B_eff += 1

    if B_eff == 0:
        raise RuntimeError("All DAG-GNN bootstraps failed; no consensus computed.")

    # ---------- probabilities ----------
    skel_prob = skel_counts.astype(float) / B_eff
    same_dir_cond = np.full((p, p), np.nan, dtype=float)
    for a in range(p):
        for b in range(a + 1, p):
            denom = skel_counts[a, b]
            if denom > 0:
                same_dir_cond[a, b] = dir_counts[a, b] / denom     # a->b | pair
                same_dir_cond[b, a] = dir_counts[b, a] / denom     # b->a | pair

    # ---------- consensus graph (GeneralGraph) ----------
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

    # ---------- save consensus PNG ----------
    png_consensus = outdir / "daggnn_boot_consensus.png"
    if not try_graphviz_png(consensus, labels, png_consensus):
        draw_networkx_fallback(consensus, labels, "DAG-GNN bootstrap consensus", png_consensus)
    print(f"[DAGGNN-BOOT] Saved: {png_consensus.name}")

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

        png_annot = outdir / "daggnn_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[DAGGNN-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[DAGGNN-BOOT] Annotation failed: {e}")

    print(f"[DAGGNN-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[DAGGNN-BOOT] Outputs in: {outdir}")

if __name__ == "__main__":
    main()
