# methods/gfci_boot.py
import argparse
import pathlib
import math
import itertools as it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime

# --- your FGES (module at fges/fges_impl.py) ---
from fges_impl import FGES

# ====== local Gaussian BIC score (interface FGES expects) ======
class GaussianBICScore:
    def __init__(self, X: np.ndarray, penalty: float = 2.0):
        self.X = X.astype(float)
        self.n, self.d = X.shape
        self.penalty = penalty

    def _bic_local(self, y: int, parents: list[int]) -> float:
        n = self.n
        yv = self.X[:, y]
        if len(parents) == 0:
            s2 = float(np.mean(yv ** 2))
            return -n * math.log(s2 + 1e-12) - self.penalty * 1 * math.log(n)
        A = self.X[:, parents]
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        resid = yv - A @ beta
        s2 = float(np.mean(resid ** 2))
        k = len(parents) + 1
        return -n * math.log(s2 + 1e-12) - self.penalty * k * math.log(n)

    def local_score_diff(self, y: int, x: int) -> float:
        return self._bic_local(y, [x]) - self._bic_local(y, [])

    def local_score_diff_parents(self, x: int, y: int, parent_indices: list[int]) -> float:
        Pa = list(parent_indices)
        if x in Pa:
            return 0.0
        return self._bic_local(y, Pa + [x]) - self._bic_local(y, Pa)

# ====== conditional-independence utils (Fisher-Z via partial corr) ======
def _normal_cdf(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def _fisher_z_pvalue(r, n, k):
    r = max(min(float(r), 0.999999), -0.999999)
    dof = n - k - 3
    if dof <= 0: return 1.0
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
    return 2.0 * (1.0 - _normal_cdf(abs(z)))

def partial_corr_pvalue(X, i, j, S_idx):
    xi = X[:, i]; xj = X[:, j]; n = X.shape[0]
    if len(S_idx) == 0:
        r = np.corrcoef(xi, xj)[0, 1]
        return _fisher_z_pvalue(r, n, 0)
    Z = X[:, S_idx]
    beta_i, *_ = np.linalg.lstsq(Z, xi, rcond=None)
    beta_j, *_ = np.linalg.lstsq(Z, xj, rcond=None)
    ri = xi - Z @ beta_i
    rj = xj - Z @ beta_j
    r = np.corrcoef(ri, rj)[0, 1]
    if np.isnan(r): return 1.0
    return _fisher_z_pvalue(r, n, len(S_idx))

# ====== PAG mark helpers: marks[i][j] in {'', 'o', '-', '>'} ======
def init_marks_from_skeleton(S):
    d = S.shape[0]
    marks = [['' for _ in range(d)] for __ in range(d)]
    for i in range(d):
        for j in range(i+1, d):
            if S[i, j] == 1:
                marks[i][j] = 'o'; marks[j][i] = 'o'
    return marks

def set_arrowhead_at(marks, i, j):  # i *-> j (arrow at j)
    if marks[i][j] != '': marks[i][j] = '>'

def set_tail_at(marks, i, j):       # i -* j (tail at j)
    if marks[i][j] != '': marks[i][j] = '-'

def has_edge(marks, i, j):
    return marks[i][j] != '' or marks[j][i] != ''

def is_circle_pair(marks, i, j):
    return marks[i][j] == 'o' and marks[j][i] == 'o'

def to_directed_adjacency_from_marks(marks):
    d = len(marks)
    D = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(d):
            if marks[i][j] == '>': D[i, j] = 1
    return D

def acyclic_if_add_arrow(marks, u, v):
    D = to_directed_adjacency_from_marks(marks)
    if D[u, v] == 1: return True
    D[u, v] = 1
    n = D.shape[0]; visited = [0]*n; stack = [0]*n
    def dfs(x):
        visited[x] = 1; stack[x] = 1
        for y in np.where(D[x]==1)[0]:
            if not visited[y]:
                if dfs(y): return True
            elif stack[y]:
                return True
        stack[x] = 0; return False
    for node in range(n):
        if not visited[node]:
            if dfs(node): return False
    return True

def snapshot(marks):
    return tuple(tuple(r) for r in marks)

# ====== FGES seeding ======
def fges_seed_skeleton_and_dir(X, penalty=2.0):
    Xs = X.astype(float)
    Xs = (Xs - Xs.mean(axis=0)) / np.where(Xs.std(axis=0)==0.0, 1.0, Xs.std(axis=0))
    d = Xs.shape[1]
    variables = list(range(d))
    score = GaussianBICScore(Xs, penalty=penalty)
    fges = FGES(variables=variables, score=score, knowledge=None, verbose=False)
    result = fges.search()
    G = result["graph"]  # networkx.DiGraph
    adj_dir = np.zeros((d, d), dtype=int)
    for u, v in G.edges(): adj_dir[u, v] = 1
    S0 = ((adj_dir + adj_dir.T) > 0).astype(int)
    return adj_dir, S0

# ====== FCI-style skeleton pruning ======
def fci_skeleton(X, S_init, alpha=0.01, k_max=3, verbose=False):
    d = S_init.shape[0]
    S = S_init.copy().astype(int)
    sep_sets = {}
    l = 0; changed = True
    while changed and l <= k_max:
        changed = False
        edges = [(i, j) for i in range(d) for j in range(i+1, d) if S[i, j] == 1]
        for (i, j) in edges:
            nbrs = set(np.where(S[i] == 1)[0]).union(set(np.where(S[j] == 1)[0]))
            nbrs.discard(i); nbrs.discard(j)
            if len(nbrs) < l: continue
            for cond in it.combinations(nbrs, l):
                p = partial_corr_pvalue(X, i, j, list(cond))
                if p > alpha:
                    S[i, j] = S[j, i] = 0
                    sep_sets[tuple(sorted((i, j)))] = set(cond)
                    changed = True
                    if verbose:
                        print(f"[Skeleton] remove {i}-{j}, sep={cond}, p={p:.3g}")
                    break
        l += 1
    return S, sep_sets

# ====== Orientations on marks ======
def orient_colliders_and_meek_on_marks(marks, sep_sets, verbose=False):
    d = len(marks)

    def adjacent(i, j): return has_edge(marks, i, j)

    changed = True
    while changed:
        changed = False
        # Colliders: i - k - j (no i-j), k not in sep(i,j) => i *-> k <-* j
        for k in range(d):
            nbrs = [u for u in range(d) if u != k and adjacent(k, u)]
            if len(nbrs) < 2: continue
            for i, j in it.combinations(nbrs, 2):
                if adjacent(i, j): continue  # shielded
                sep = sep_sets.get(tuple(sorted((i, j))), set())
                if k not in sep:
                    if adjacent(i, k) and marks[i][k] != '>' and acyclic_if_add_arrow(marks, i, k):
                        set_arrowhead_at(marks, i, k); changed = True
                    if adjacent(j, k) and marks[j][k] != '>' and acyclic_if_add_arrow(marks, j, k):
                        set_arrowhead_at(marks, j, k); changed = True

        # Meek R2: A *-> B and B o- C and A not adj C => B -> C
        Dtmp = to_directed_adjacency_from_marks(marks)
        for A in range(d):
            for B in range(d):
                if Dtmp[A, B] != 1: continue
                for C in range(d):
                    if C in (A, B): continue
                    if (not has_edge(marks, A, C)) and has_edge(marks, B, C):
                        if marks[B][C] == 'o' and marks[C][B] == 'o':
                            if acyclic_if_add_arrow(marks, B, C):
                                set_tail_at(marks, B, C)
                                set_arrowhead_at(marks, B, C)
                                set_tail_at(marks, C, B)
                                changed = True

        # Meek R3: A o- B and path A -> ... -> B => A -> B
        Dtmp = to_directed_adjacency_from_marks(marks)
        def has_path(src, dst, seen=None):
            if seen is None: seen = set()
            if src == dst: return True
            seen.add(src)
            for child in np.where(Dtmp[src] == 1)[0]:
                if child not in seen and has_path(child, dst, seen):
                    return True
            return False

        for A in range(d):
            for B in range(d):
                if A == B: continue
                if has_edge(marks, A, B) and marks[A][B] == 'o' and marks[B][A] == 'o':
                    if has_path(A, B) and acyclic_if_add_arrow(marks, A, B):
                        set_tail_at(marks, A, B)
                        set_arrowhead_at(marks, A, B)
                        set_tail_at(marks, B, A)
                        changed = True

    return marks

# ---- FGES adoption (strong) + half-arrow completion, both gated by FGES skeleton S0
def adopt_fges_on_marks_strong(marks, adj_fges_dir, S0):
    """
    Adopt FGES direction only if the pair (i,j) existed in the FGES skeleton S0.
    Works for o-o and o-> / <-o, and stays acyclic.
    """
    d = len(marks); changed = False
    for i in range(d):
        for j in range(i+1, d):
            if not has_edge(marks, i, j): continue
            if S0[i, j] == 0: continue
            L, R = marks[i][j], marks[j][i]
            if (L == '>' and R == '-') or (L == '-' and R == '>'):
                continue
            if adj_fges_dir[i, j] == 1 and R != '>' and acyclic_if_add_arrow(marks, i, j):
                set_tail_at(marks, i, j); set_arrowhead_at(marks, i, j); set_tail_at(marks, j, i)
                changed = True; continue
            if adj_fges_dir[j, i] == 1 and L != '>' and acyclic_if_add_arrow(marks, j, i):
                set_tail_at(marks, j, i); set_arrowhead_at(marks, j, i); set_tail_at(marks, i, j)
                changed = True
    return changed

def complete_half_arrows(marks, S0):
    """
    Only complete (o->) or (<-o) into (->) if the pair was in the FGES skeleton.
    """
    d = len(marks); changed = False
    for i in range(d):
        for j in range(i+1, d):
            if S0[i, j] == 0: continue
            L, R = marks[i][j], marks[j][i]
            if L == '>' and R == 'o':
                set_tail_at(marks, j, i); changed = True
            elif R == '>' and L == 'o':
                set_tail_at(marks, i, j); changed = True
    return changed

def closure_to_convergence(marks, sep_sets, adj_fges_dir, S0,
                           use_fges=True, complete_half=True):
    changed = True
    while changed:
        before = snapshot(marks)
        orient_colliders_and_meek_on_marks(marks, sep_sets, verbose=False)
        if use_fges:
            _ = adopt_fges_on_marks_strong(marks, adj_fges_dir, S0)
        if complete_half:
            _ = complete_half_arrows(marks, S0)
        changed = (snapshot(marks) != before)
    return marks

# ====== causallearn Graph for nice Graphviz rendering ======
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils
import networkx as nx  # fallback only

def marks_to_generalgraph(marks, labels):
    m2ep = {'o': Endpoint.CIRCLE, '-': Endpoint.TAIL, '>': Endpoint.ARROW, '': None}
    nodes = [GraphNode(lbl) for lbl in labels]
    GG = GeneralGraph(nodes)
    p = len(labels)
    for i in range(p):
        for j in range(i+1, p):
            if not has_edge(marks, i, j): continue
            epL = m2ep[marks[i][j]]
            epR = m2ep[marks[j][i]]
            if epL is None or epR is None:
                continue
            GG.add_edge(Edge(nodes[i], nodes[j], epL, epR))
    return GG

def try_graphviz_png(GG: GeneralGraph, labels, out_png: pathlib.Path) -> bool:
    try:
        pyd = GraphUtils.to_pydot(GG, labels=labels)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pyd.write_png(str(out_png)); return True
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
                edges = [e]; break
    for e in edges:
        e.set("xlabel", text)
        e.set("labelfontsize", "10")
        e.set("labelfontcolor", "black")
        e.set("labeldistance", "1.6")
        e.set("labelangle", "0")

# ====== fixed internal settings (no CLI exposure) ======
ALPHA    = 0.01   # Fisher-Z alpha in pruning
K_MAX    = 4      # max cond set size in pruning (match your liked version)
PENALTY  = 2.0    # FGES BIC penalty
EDGE_THR = 0.50   # include pair in consensus if skeleton prob ≥ this
DIR_THR  = 0.50   # orient if conditional same-dir prob ≥ this AND ≥ opposite
RNG_SEED = 12345

# ====== run a single GFCI pass → marks ======
def run_gfci_marks(X):
    # 1) FGES seed
    adj_fges_dir, S0 = fges_seed_skeleton_and_dir(X, penalty=PENALTY)
    # 2) FCI-style skeleton pruning
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0)==0.0, 1.0, X.std(axis=0))
    Spruned, sep_sets = fci_skeleton(Xs, S0, alpha=ALPHA, k_max=K_MAX, verbose=False)
    # 3) init PAG marks
    marks = init_marks_from_skeleton(Spruned)
    # 4) closure: colliders+Meek → strong FGES adoption (S0-gated) → half-arrow completion (S0-gated)
    marks = closure_to_convergence(marks, sep_sets, adj_fges_dir, S0,
                                   use_fges=True, complete_half=True)
    return marks

# ====== main (bootstrap + consensus) ======
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
    df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns after cleaning.")
    labels = df.columns.tolist()
    X = df.to_numpy(dtype=float)
    n, p = X.shape

    print(f"[GFCI-BOOT] Data {X.shape} | n_boot={args.n_boot} | alpha={ALPHA}, k_max={K_MAX}, penalty={PENALTY}")

    # bootstrap tallies
    B_eff = 0
    skel_counts = np.zeros((p, p), dtype=int)          # unordered (a<b): pair present in final PAG
    dir_counts  = np.zeros((p, p), dtype=int)          # ordered: i->j (fully directed only, in final PAG)
    oriented_pair_counts = np.zeros((p, p), dtype=int) # unordered: pair was fully oriented (-> or <-) in final PAG
    counts_by_pair = defaultdict(Counter)              # unordered: endpoint-pattern histogram (final PAG)

    rng = np.random.default_rng(RNG_SEED)
    for b in range(args.n_boot):
        rows = rng.integers(0, n, size=n, endpoint=False)
        Xb = X[rows, :]
        try:
            marks = run_gfci_marks(Xb)
        except Exception as e:
            print(f"[GFCI-BOOT] replicate {b}: FAILED -> {e}")
            continue

        present_pairs = set()
        for i in range(p):
            for j in range(i+1, p):
                if not has_edge(marks, i, j):
                    continue

                a, bpair = (i, j)  # i<j by construction
                present_pairs.add((a, bpair))  # presence in final PAG (after step 4)

                # Final endpoint pattern (after step 4)
                L, R = marks[i][j], marks[j][i]
                counts_by_pair[(a, bpair)][(L, R)] += 1

                # Direction counts (only fully directed -> or <- in final PAG)
                if L == '>' and R == '-':
                    dir_counts[i, j] += 1
                    oriented_pair_counts[a, bpair] += 1
                elif L == '-' and R == '>':
                    dir_counts[j, i] += 1
                    oriented_pair_counts[a, bpair] += 1

        # Skeleton presence (unordered) after final PAG
        for (a, bpair) in present_pairs:
            skel_counts[a, bpair] += 1

        B_eff += 1

    if B_eff == 0:
        raise RuntimeError("All GFCI bootstraps failed; no consensus computed.")

    # Skeleton probability over all successful bootstraps (unordered)
    skel_prob = skel_counts.astype(float) / B_eff

    # Direction probability conditional on being fully oriented (unordered)
    # P(i -> j | pair fully oriented in final PAG)
    dir_cond_oriented = np.full((p, p), np.nan, dtype=float)
    for a in range(p):
        for b in range(a+1, p):
            denom_oriented = oriented_pair_counts[a, b]
            if denom_oriented > 0:
                dir_cond_oriented[a, b] = dir_counts[a, b] / denom_oriented
                dir_cond_oriented[b, a] = dir_counts[b, a] / denom_oriented

    # consensus marks
    consensus = [['' for _ in range(p)] for __ in range(p)]

    def set_pair(i, j, L, R):
        consensus[i][j] = L
        consensus[j][i] = R

    for a in range(p):
        for b in range(a+1, p):
            sp = skel_prob[a, b]
            if sp < EDGE_THR:
                continue

            pab = 0.0 if np.isnan(dir_cond_oriented[a, b]) else dir_cond_oriented[a, b]
            pba = 0.0 if np.isnan(dir_cond_oriented[b, a]) else dir_cond_oriented[b, a]

            if pab >= pba and pab >= DIR_THR:
                set_pair(a, b, '>', '-')
                continue
            if pba > pab and pba >= DIR_THR:
                set_pair(a, b, '-', '>')
                continue

            # ambiguous: choose most frequent ambiguous pattern (exclude clear '>-'/ '->')
            hist = counts_by_pair.get((a, b), Counter())
            def is_amb(k):
                L, R = k
                return not ((L == '>' and R == '-') or (L == '-' and R == '>'))
            amb = {k: v for k, v in hist.items() if is_amb(k)}
            if amb:
                (L, R), _ = max(amb.items(), key=lambda kv: kv[1])
                set_pair(a, b, L, R)
            else:
                set_pair(a, b, 'o', 'o')

    # render via causallearn -> graphviz
    GG = marks_to_generalgraph(consensus, labels)
    png_consensus = outdir / "gfci_boot_consensus.png"
    if try_graphviz_png(GG, labels, png_consensus):
        print(f"[GFCI-BOOT] Saved: {png_consensus.name}")
    else:
        # minimal fallback (directed & undirected only)
        try:
            Gd = nx.DiGraph()
            Gd.add_nodes_from(labels)
            und = []
            for i in range(p):
                for j in range(i+1, p):
                    L, R = consensus[i][j], consensus[j][i]
                    if L == '>' and R == '-':
                        Gd.add_edge(labels[i], labels[j])
                    elif L == '-' and R == '>':
                        Gd.add_edge(labels[j], labels[i])
                    elif L == '-' and R == '-':
                        und.append((labels[i], labels[j]))
            pos = nx.spring_layout(Gd, seed=42)
            plt.figure(figsize=(11, 8))
            nx.draw(Gd, pos, with_labels=True, node_color="#E6F0FF", node_size=900, edgecolors="#345")
            if und:
                Gu = nx.Graph(); Gu.add_nodes_from(labels); Gu.add_edges_from(und)
                nx.draw_networkx_edges(Gu, pos, edgelist=und, style="dashed", width=1.5)
            plt.title("GFCI bootstrap consensus (fallback)")
            plt.tight_layout()
            plt.savefig(png_consensus, dpi=150)
            plt.close()
            print(f"[GFCI-BOOT] Saved (fallback): {png_consensus.name}")
        except Exception as e:
            print(f"[GFCI-BOOT] Fallback plot failed: {e}")

    # ---------- annotated PNG (skeleton%, dir% | pair) ----------
    # NOTE: this runs regardless of which branch rendered the first PNG.
    try:
        pyd = GraphUtils.to_pydot(GG, labels=labels)
        label2id = pydot_label_map(pyd)

        for e in GG.get_graph_edges():
            n1 = e.get_node1().get_name()
            n2 = e.get_node2().get_name()
            i = labels.index(n1)
            j = labels.index(n2)
            a, b = (i, j) if i < j else (j, i)

            sk = skel_prob[a, b]  # P(pair present) over bootstraps
            ep1 = e.get_endpoint1().value
            ep2 = e.get_endpoint2().value

            if ep1 == Endpoint.TAIL.value and ep2 == Endpoint.ARROW.value:
                # i -> j
                pd_or = dir_cond_oriented[i, j]      # P(i->j | pair fully oriented)
                dir_txt = "—" if np.isnan(pd_or) else f"{pd_or*100:.0f}%"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, {dir_txt})")

            elif ep1 == Endpoint.ARROW.value and ep2 == Endpoint.TAIL.value:
                # j -> i (edge stored as n1–n2)
                pd_or = dir_cond_oriented[j, i]      # P(j->i | pair fully oriented)
                dir_txt = "—" if np.isnan(pd_or) else f"{pd_or*100:.0f}%"
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, {dir_txt})")

            else:
                # ambiguous endpoints (o-o, o->, <-o, --, <->): show only skeleton %
                set_edge_xlabel(pyd, label2id[n1], label2id[n2], f"({sk*100:.0f}%, —)")

        png_annot = outdir / "gfci_boot_consensus_annotated.png"
        pyd.write_png(str(png_annot))
        print(f"[GFCI-BOOT] Saved: {png_annot.name}")
    except Exception as e:
        print(f"[GFCI-BOOT] Annotation failed: {e}")

    # optional: save raw arrays
    np.save(outdir / "gfci_boot_skeleton_prob.npy", skel_prob)
    np.save(outdir / "gfci_boot_dir_prob.npy", dir_cond_oriented)

    print(f"[GFCI-BOOT] Bootstraps used: {B_eff}/{args.n_boot}")
    print(f"[GFCI-BOOT] Outputs in: {outdir}")


if __name__ == "__main__":
    main()
