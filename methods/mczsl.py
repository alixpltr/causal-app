# -*- coding: utf-8 -*-
"""
MCSL (gCastle) — learn binary DAG, print it, and SHOW the plot (no saving).
- No scikit-learn needed (NumPy z-score).
- Tuned to be less sparse (more edges) via graph_thresh ↓ and l1_graph_penalty ↓.
"""

import numpy as np
import pandas as pd
from castle.algorithms import MCSL

# plotting (interactive)
import networkx as nx
import matplotlib.pyplot as plt

# show full matrices in console
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def zscore_numpy(X: np.ndarray) -> np.ndarray:
    X = X.astype(float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd

def plot_dag_show(A_df: pd.DataFrame, colnames, title="MCSL – Binary DAG"):
    """Show a DAG window from binary adjacency A_df (0/1)."""
    G = nx.DiGraph()
    G.add_nodes_from(colnames)

    # add edges
    for i, src in enumerate(colnames):
        row = A_df.iloc[i].to_numpy()
        for j, val in enumerate(row):
            if val == 1 and src != colnames[j]:
                G.add_edge(src, colnames[j])

    if G.number_of_edges() == 0:
        print("DAG has no edges to plot. Try lowering graph_thresh or l1_graph_penalty.")
        return

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.6, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=8)
    # robust edge drawing (avoids rare matplotlib glitches)
    try:
        nx.draw_networkx_edges(
            G, pos, arrows=True, arrowstyle="->", arrowsize=15, connectionstyle="arc3,rad=0.05"
        )
    except Exception as e:
        print(f"[edge draw warning] {e} — falling back to simple edges.")
        nx.draw_networkx_edges(G, pos, arrows=False)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    # your data path
    csv_path = r"C:\Users\jciccoli\Desktop\Projetcausal\alix\data\signature_multimpo.csv"

    # load & preprocess (numeric columns only; drop NaN/Inf)
    df = pd.read_csv(csv_path)
    X_df = (df.select_dtypes(include=[np.number])
              .replace([np.inf, -np.inf], np.nan)
              .dropna(axis=0))
    colnames = X_df.columns.tolist()
    X = zscore_numpy(X_df.to_numpy())

    # --- MCSL (less-sparse preset) ---
    # Based on gCastle MCSL defaults + demo knobs; relaxed to get more edges.
    model = MCSL(
        model_type='nn',
        num_hidden_layers=4,
        hidden_dim=32,           # ↑ capacity
        graph_thresh=0.4,       # ↓ threshold → more edges (try 0.25 if still sparse)
        l1_graph_penalty=5e-3,   # ↓ sparsity → more edges (try 5e-4 for even denser)
        learning_rate=3e-2,
        max_iter=30,             # outer AL iterations (25 default)
        iter_step=800,           # steps per iter (1000 default)
        init_iter=2,
        h_tol=1e-10,
        init_rho=1e-5,
        rho_thresh=1e18,         # allow larger penalty growth (1e14 default)
        h_thresh=0.30,           # slightly looser than default 0.25
        rho_multiply=10.0,
        temperature=0.2,
        device_type='cpu',       # set 'gpu' if CUDA is OK for you
        device_ids='0',
        random_seed=42
    )
    model.learn(X, columns=colnames)

    # get binary adjacency (MCSL learns binary by design)
    if hasattr(model, "causal_matrix"):
        A = np.asarray(model.causal_matrix).astype(int)
    elif hasattr(model, "model") and hasattr(model.model, "adjacency"):
        A = model.model.adjacency.detach().cpu().numpy().astype(int)
    else:
        raise AttributeError("No adjacency found on MCSL object.")

    A_df = pd.DataFrame(A, index=colnames, columns=colnames)

    # print DAG and edge count
    print("\n=== Binary DAG adjacency (rows → cols) ===")
    print(A_df)
    print(f"\nTotal edges: {int(A.sum())}")

    # show plot
    plot_dag_show(A_df, colnames, title="MCSL – Binary DAG (less-sparse preset)")

if __name__ == "__main__":
    main()
