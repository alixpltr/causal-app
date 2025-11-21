import os
import datetime
import numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd.extend import primitive, defvjp
import functools
import scipy.optimize as sopt
import pandas as pd
from scipy.special import comb
import copy
import matplotlib.pyplot as plt

# =========================
# Local ADMG (minimal API)
# =========================
class ADMG:
    def __init__(self, vertices):
        self.vertices = list(vertices)
        self.di_edges = []   # list of (u, v) directed edges u -> v
        self.bi_edges = []   # list of (u, v) bidirected edges (stored once; symmetric)

    def add_diedge(self, u, v):
        if (u, v) not in self.di_edges:
            self.di_edges.append((u, v))

    def add_biedge(self, u, v):
        if u == v:
            return
        if (u, v) not in self.bi_edges and (v, u) not in self.bi_edges:
            self.bi_edges.append((u, v))

    def has_biedge(self, u, v):
        return (u, v) in self.bi_edges or (v, u) in self.bi_edges

    def __repr__(self):
        return f"ADMG(vertices={len(self.vertices)}, di_edges={len(self.di_edges)}, bi_edges={len(self.bi_edges)})"


# =========================
# Cycle loss w/ autograd VJP
# =========================
@primitive
def cycle_loss(W):
    d = len(W)
    M = np.eye(d) + W * W / d
    E = np.linalg.matrix_power(M, d - 1)
    return (E.T * M).sum() - d

def dcycle_loss(ans, W):
    W_shape = W.shape
    d = len(W)
    M = anp.eye(d) + W * W / d
    E = anp.linalg.matrix_power(M, d - 1)
    return lambda g: anp.full(W_shape, g) * E.T * W * 2

defvjp(cycle_loss, dcycle_loss)


# =========================
# Structure penalties
# =========================
def ancestrality_loss(W1, W2):
    d = len(W1)
    W1_pos = anp.multiply(W1, W1)
    W2_pos = anp.multiply(W2, W2)
    W1k = np.eye(d)
    M = np.eye(d)
    for k in range(1, d):
        W1k = anp.dot(W1k, W1_pos)
        M += 1.0 / np.math.factorial(k) * W1k
    return anp.sum(anp.multiply(M, W2_pos))

def reachable_loss(W1, W2, alpha_d=1, alpha_b=2, s=anp.log(5000)):
    d = len(W1)
    greenery = 0
    for var_index in range(d):
        mask = anp.array([1 if i == var_index else 0 for i in range(d)]) * 1
        W1_fixed = anp.multiply(W1, W1)
        W2_fixed = anp.multiply(W2, W2)
        for _ in range(d - 1):
            Bk = np.eye(d)
            M = np.eye(d)
            for k in range(1, d):
                Bk = anp.dot(Bk, W2_fixed)
                M += comb(d, k) * (alpha_b ** k) * Bk
            p_fixability_matrix = anp.multiply(M, W1_fixed)
            e2x = anp.exp(anp.clip(s * (anp.mean(p_fixability_matrix, axis=1) + mask), 0, 4))
            fixability = (e2x - 1) / (e2x + 1)
            fixability_mask = anp.vstack([fixability for _ in range(d)])
            W1_fixed = anp.multiply(W1_fixed, fixability_mask)
            W2_fixed = anp.multiply(W2_fixed, fixability_mask)
            W2_fixed = anp.multiply(W2_fixed, fixability_mask.T)
        Bk, Dk = np.eye(d), np.eye(d)
        eW1_fixed, eW2_fixed = np.eye(d), np.eye(d)
        for k in range(1, d):
            Dk = anp.dot(Dk, W1_fixed)
            Bk = anp.dot(Bk, W2_fixed)
            eW1_fixed += 1 / np.math.factorial(k) * Dk
            eW2_fixed += 1 / np.math.factorial(k) * Bk
            eW1_fixed += comb(d, k) * (alpha_d ** k) * Dk
            eW2_fixed += comb(d, k) * (alpha_b ** k) * Bk
        greenery += anp.sum(anp.multiply(eW1_fixed[:, var_index], eW2_fixed[:, var_index])) - 1
    return greenery

def bow_loss(W1, W2):
    W1_pos = anp.multiply(W1, W1) / len(W1)
    W2_pos = anp.multiply(W2, W2) / len(W1)
    return anp.sum(anp.multiply(W1_pos, W2_pos))


# =========================
# Discovery class (patched)
# =========================
class Discovery:
    def __init__(self, lamda=0.05):
        self.X_ = None
        self.S_ = None
        self.Z_ = None
        self.W1_ = None
        self.W2_ = None
        self.Wii_ = None
        self.convergence_ = None
        self.lamda = lamda
        self.G_ = None
        self.last_sigma_ = None   # model-implied covariance for the final solution

    def primal_loss(self, params, rho, alpha, Z, structure_penalty_func):
        n, d = self.X_.shape
        W1 = anp.reshape(params[0:d * d], (d, d))
        W2 = anp.reshape(params[d * d:], (d, d))
        W2 = W2 + W2.T

        loss = 0.0
        for var_index in range(d):
            loss += 0.5 / n * anp.linalg.norm(
                self.X_[:, var_index]
                - anp.dot(self.X_, W1[:, var_index])
                - anp.dot(Z[var_index], W2[:, var_index])
            ) ** 2

        structure_penalty = cycle_loss(W1) + structure_penalty_func(W1, W2)
        structure_penalty = 0.5 * rho * (structure_penalty ** 2) + alpha * structure_penalty
        eax2 = anp.exp((anp.log(n) * anp.abs(params)))
        tanh = (eax2 - 1) / (eax2 + 1)
        return loss + structure_penalty + anp.sum(tanh) * self.lamda

    def _create_bounds(self, tiers, unconfounded_vars, var_names):
        if tiers is None:
            tiers = [var_names]
        unconfounded_vars = set(unconfounded_vars)
        tier_dict = {}
        for tier_num in range(len(tiers)):
            for var in tiers[tier_num]:
                tier_dict[var] = tier_num
        bounds_directed_edges = []
        bounds_bidirected_edges = []
        for i in range(len(var_names)):
            for j in range(len(var_names)):
                if i == j:
                    bounds_directed_edges.append((0, 0))
                elif tier_dict[var_names[i]] > tier_dict[var_names[j]]:
                    bounds_directed_edges.append((0, 0))
                else:
                    bounds_directed_edges.append((-4, 4))
                if i <= j:
                    bounds_bidirected_edges.append((0, 0))
                elif var_names[i] in unconfounded_vars or var_names[j] in unconfounded_vars:
                    bounds_bidirected_edges.append((0, 0))
                else:
                    bounds_bidirected_edges.append((-4, 4))
        return bounds_directed_edges + bounds_bidirected_edges

    def _compute_pseudo_variables(self, W1, W2):
        Z = {}
        d = len(W1)
        for var_index in range(d):
            indices = list(range(0, var_index)) + list(range(var_index + 1, d))
            omega_minusii = W2[anp.ix_(indices, indices)]
            omega_minusii_inv = anp.linalg.inv(omega_minusii)
            epsilon = self.X_ - anp.matmul(self.X_, W1)
            epsilon_minusi = anp.delete(epsilon, var_index, axis=1)
            Z_minusi = (omega_minusii_inv @ epsilon_minusi.T).T
            Z[var_index] = anp.insert(Z_minusi, var_index, 0, axis=1)
        return Z

    def get_graph(self, W1, W2, vertices, threshold):
        G = ADMG(vertices)
        for i in range(len(vertices)):
            for j in range(len(vertices)):
                if abs(W1[i, j]) > threshold:
                    G.add_diedge(vertices[i], vertices[j])
                if i != j and abs(W2[i, j]) > threshold and not G.has_biedge(vertices[i], vertices[j]):
                    G.add_biedge(vertices[i], vertices[j])
        return G

    def _discover_admg(self, data, admg_class, tiers=None, unconfounded_vars=[], max_iter=100,
                       h_tol=1e-8, rho_max=1e+16, w_threshold=0.05,
                       ricf_increment=1, ricf_tol=1e-4, verbose=False):
        self.X_ = anp.copy(data.values)
        n, d = self.X_.shape
        self.S_ = anp.cov(self.X_.T)

        bounds = self._create_bounds(tiers, unconfounded_vars, data.columns)

        W1_hat = anp.random.uniform(-0.5, 0.5, (d, d))
        W2_hat = anp.random.uniform(-0.05, 0.05, (d, d))
        W2_hat[np.tril_indices(d)] = 0
        W2_hat = W2_hat + W2_hat.T
        W2_hat = anp.multiply(W2_hat, 1 - np.eye(d))
        Wii_hat = anp.diag(anp.diag(self.S_))

        rho, alpha, h = 1.0, 0.0, np.inf
        ricf_max_iters = 1
        convergence = False

        if admg_class == "ancestral":
            penalty = ancestrality_loss
        elif admg_class == "arid":
            penalty = reachable_loss
        elif admg_class == "bowfree":
            penalty = bow_loss
        else:
            raise NotImplementedError("Invalid ADMG class")

        objective = functools.partial(self.primal_loss)
        gradient = grad(objective)

        for num_iter in range(max_iter):
            W1_new, W2_new, Wii_new = None, None, None
            h_new = None
            while rho < rho_max:
                W1_new, W2_new, Wii_new = W1_hat.copy(), W2_hat.copy(), Wii_hat.copy()
                ricf_iter = 0
                while ricf_iter < ricf_max_iters:
                    ricf_iter += 1
                    W1_old = W1_new.copy()
                    W2_old = W2_new.copy()
                    Wii_old = Wii_new.copy()

                    Z = self._compute_pseudo_variables(W1_new, W2_new + Wii_new)
                    current_estimates = np.concatenate((W1_new.flatten(), W2_new.flatten()))
                    sol = sopt.minimize(self.primal_loss, current_estimates,
                                        args=(rho, alpha, Z, penalty),
                                        method='L-BFGS-B',
                                        options={'disp': False}, bounds=bounds, jac=gradient)

                    W1_new = np.reshape(sol.x[0:d * d], (d, d))
                    W2_new = np.reshape(sol.x[d * d:], (d, d))
                    W2_new = W2_new + W2_new.T

                    for var_index in range(d):
                        Wii_new[var_index, var_index] = np.var(
                            self.X_[:, var_index] - np.dot(self.X_, W1_new[:, var_index]))

                    if np.sum(np.abs(W1_old - W1_new)) + np.sum(np.abs((W2_old + Wii_old) - (W2_new + Wii_new))) < ricf_tol:
                        convergence = True
                        break

                h_new = cycle_loss(W1_new) + penalty(W1_new, W2_new)
                if verbose:
                    print(num_iter, float(h_new))
                if h_new < 0.25 * h:
                    break
                else:
                    rho *= 10

            W1_hat, W2_hat, Wii_hat = W1_new.copy(), W2_new.copy(), Wii_new.copy()
            h = h_new
            alpha += rho * h
            ricf_max_iters += ricf_increment
            if h <= h_tol or rho >= rho_max:
                break

        final_W1, final_W2 = W1_hat.copy(), W2_hat + Wii_hat
        # keep for scoring
        self.W1_, self.W2_, self.Wii_ = W1_hat.copy(), W2_hat.copy(), Wii_hat.copy()

        final_W1[np.abs(final_W1) < w_threshold] = 0
        final_W2[np.abs(final_W2) < w_threshold] = 0
        G = self.get_graph(final_W1, final_W2, data.columns, w_threshold)
        # store model-implied covariance (for BIC)
        I = np.eye(d)
        try:
            A = np.linalg.inv(I - W1_hat)
            Sigma = A @ (W2_hat + Wii_hat) @ A.T
        except np.linalg.LinAlgError:
            Sigma = None
        self.last_sigma_ = Sigma
        return G, convergence, (W1_hat.copy(), (W2_hat + Wii_hat).copy())

    # ---------- Local BIC scorer (Gaussian) ----------
    def _bic_from_params(self, data: pd.DataFrame, W1: np.ndarray, Omega: np.ndarray, threshold=1e-8):
        """
        BIC = -2 loglik + k log n for linear Gaussian SEM
        Sigma = (I - W1)^{-1} Omega (I - W1)^{-T}
        k ~ number of free parameters: nonzero W1 entries + nonzero off-diag Omega + d diag variances
        """
        X = data.values
        n, d = X.shape
        S = np.cov(X.T)
        I = np.eye(d)
        try:
            A = np.linalg.inv(I - W1)
            Sigma = A @ Omega @ A.T
            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                return np.inf
            Sigma_inv = np.linalg.inv(Sigma)
            ll = -0.5 * n * (d * np.log(2 * np.pi) + logdet + np.trace(S @ Sigma_inv))
            # parameter count
            k_w1 = (np.abs(W1) > threshold).sum()
            offdiag = np.triu(np.ones((d, d)), 1).astype(bool)
            k_omega_off = (np.abs(Omega[offdiag]) > threshold).sum()
            k_omega_diag = d  # diagonal variances (Wii)
            k = int(k_w1 + k_omega_off + k_omega_diag)
            bic = -2 * ll + k * np.log(n)
            return float(bic)
        except np.linalg.LinAlgError:
            return np.inf

    def discover_admg(self, data, admg_class, tiers=None, unconfounded_vars=[], max_iter=100,
                      h_tol=1e-8, rho_max=1e+16, num_restarts=5, w_threshold=0.05,
                      ricf_increment=1, ricf_tol=1e-4, verbose=False, detailed_output=False):
        best_bic = np.inf
        best_G = None
        best_conv = False
        for i in range(num_restarts):
            if verbose:
                print(f"=== Random restart {i+1}/{num_restarts} ===")
            G, convergence, params = self._discover_admg(
                data, admg_class, tiers, unconfounded_vars, max_iter,
                h_tol, rho_max, w_threshold, ricf_increment, ricf_tol, detailed_output
            )
            W1_hat, Omega_hat = params
            curr_bic = self._bic_from_params(data, W1_hat, Omega_hat, threshold=w_threshold/2)
            if verbose:
                print("Estimated di_edges:", G.di_edges)
                print("Estimated bi_edges:", G.bi_edges)
                print("BIC:", curr_bic)
            if curr_bic < best_bic:
                best_bic = curr_bic
                self.G_ = copy.deepcopy(G)
                self.convergence_ = convergence
                best_G = copy.deepcopy(G)
                # keep the best params
                self.W1_, self.W2_ = W1_hat, Omega_hat  # note: here W2_ stores Omega (incl diag)
        if verbose:
            print("Final estimated di_edges:", self.G_.di_edges)
            print("Final estimated bi_edges:", self.G_.bi_edges)
            print("Final BIC:", best_bic)
        return best_G


# =========================
# Runner / Saving Helpers
# =========================
def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        col = out[c].astype(float)
        std = col.std(ddof=0)
        mean = col.mean()
        out[c] = (col - mean) / std if std > 0 else (col - mean)
    return out

def make_tiers(labels, outcome=None):
    if outcome is None or outcome not in labels:
        return [labels]
    base = [v for v in labels if v != outcome]
    return [base, [outcome]]

def graph_to_adjacency(G: ADMG, labels):
    n = len(labels)
    idx = {v: i for i, v in enumerate(labels)}
    directed = np.zeros((n, n), dtype=int)
    bidirected = np.zeros((n, n), dtype=int)
    for u, v in G.di_edges:
        if u in idx and v in idx:
            directed[idx[u], idx[v]] = 1
    for u, v in G.bi_edges:
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            bidirected[i, j] = 1
            bidirected[j, i] = 1
    return directed, bidirected

def plot_adj(adj, labels, title, out_png=None):
    plt.figure(figsize=(12, 10))
    im = plt.imshow(adj, cmap='Blues', vmin=0, vmax=1)
    plt.title(title)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90)
    plt.yticks(ticks=range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, adj[i, j], ha='center', va='center', fontsize=7)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
    plt.show()