#!/usr/bin/env python3
"""
02_models.py
============
The six prediction models compared in:

    Vitale et al. "Exploring Machine- and Deep-Learning Algorithms for
    Hybrid Crop Prediction".

    GBLUP      ridge regression on the eigenvector predictors, with a
               separate variance ratio per relationship matrix
    SVR        support vector regression, linear and RBF kernels
    RF         random forest
    LightGBM   gradient boosting, leaf-wise, histogram splits
    MLP        multilayer perceptron
    AGNET      Anchored Genomic NETwork: multi-matrix GBLUP with an
               embedded neural branch (this paper)

plus the nested cross-validation used to evaluate them.

Run it as-is for a worked example on simulated data:

    python 02_models.py

Requires: numpy, scipy, scikit-learn.  torch and lightgbm are optional --
the corresponding models are skipped with a message if absent.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from importlib import import_module

# Note: 01_relationship_matrices.py cannot be imported directly because a
# Python module name may not begin with a digit. The few helpers needed here
# are therefore repeated below. Keep the two files in sync if you edit them,
# or rename the scripts (e.g. matrices.py / models.py) and import normally.


# ══════════════════════════════════════════════════════════════════════
# feature preparation
# ══════════════════════════════════════════════════════════════════════

def spearcor_topk(X, y, k):
    """Indices of the k predictors with the strongest |Spearman| with y.

    Used only for the marker representation. Must be computed on the
    TRAINING rows alone, or test phenotypes leak into feature selection.
    """
    if X.shape[1] <= k:
        return np.arange(X.shape[1])
    rx = rankdata(X, axis=0)                    # vectorised: ~50x faster
    ry = rankdata(y)
    rx = rx - rx.mean(0)
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum())
    denom[denom < 1e-12] = np.inf
    r = np.abs((rx * ry[:, None]).sum(0) / denom)
    return np.argsort(-r)[:k]


def apply_scaling(X_tr, X_te, scale_ok):
    """Standardise columns for markers; pass eigenvector predictors through.

    This guard is load-bearing. The L^(1/2) column weighting of the
    eigenvector predictors IS the GBLUP prior; standardising the columns to
    unit variance removes it. Measured on wheat: predictive ability fell from
    0.549 to -0.013 when the guard was removed.
    """
    if not scale_ok:
        return X_tr, X_te
    mu = X_tr.mean(0)
    sd = X_tr.std(0)
    sd[sd < 1e-12] = 1.0
    return (X_tr - mu) / sd, (X_te - mu) / sd


def _standardise_y(y):
    mu, sd = float(y.mean()), float(y.std())
    return (y - mu) / (sd if sd > 1e-12 else 1.0), mu, (sd if sd > 1e-12 else 1.0)


# ══════════════════════════════════════════════════════════════════════
# 1. GBLUP  (ridge on the eigenvector predictors)
# ══════════════════════════════════════════════════════════════════════
#
# Ridge on Z* = U L^(1/2) is algebraically GBLUP. With several relationship
# matrices, however, a single penalty cannot give each matrix its own
# variance component, which is what a Bayesian implementation such as BGLR
# does. Per-block weights w_b recover that: fitting on K = sum_b w_b K_b is
# equivalent to a separate variance ratio per matrix. Weight 0 lets an
# uninformative matrix be switched off entirely.

ALPHAS = np.logspace(-1, 5, 13)
BLOCK_WEIGHTS = [0.0, 1.0, 10.0]


def ridge_grid(block_sizes):
    if len(block_sizes) <= 1:
        return [{"alpha": a, "weights": None} for a in ALPHAS]
    from itertools import product
    ws = [w for w in product(BLOCK_WEIGHTS, repeat=len(block_sizes))
          if any(v > 0 for v in w)]
    return [{"alpha": a, "weights": w}
            for a in np.logspace(-1, 5, 7) for w in ws]


def ridge_fit(X_tr, y_tr, X_te, hp, block_sizes=None, seed=0):
    w = hp.get("weights")
    if w is None or block_sizes is None or len(block_sizes) <= 1:
        return Ridge(alpha=hp["alpha"]).fit(X_tr, y_tr).predict(X_te)

    mu = float(y_tr.mean())
    K_tr = np.zeros((len(X_tr), len(X_tr)))
    K_te = np.zeros((len(X_te), len(X_tr)))
    s = 0
    for b, wb in zip(block_sizes, w):
        if wb > 0:
            Ztr = X_tr[:, s:s + b]
            K_tr += wb * (Ztr @ Ztr.T)
            K_te += wb * (X_te[:, s:s + b] @ Ztr.T)
        s += b
    c = np.linalg.solve(K_tr + hp["alpha"] * np.eye(len(K_tr)), y_tr - mu)
    return K_te @ c + mu


# ══════════════════════════════════════════════════════════════════════
# 2. SVR
# ══════════════════════════════════════════════════════════════════════
#
# gamma is fixed at scikit-learn's "scale" heuristic, 1 / (p * Var(X)),
# computed within the training fold. It applies to the RBF kernel only.
# max_iter is capped: libsvm runs to convergence by default and on noisy
# genomic data with large C and small epsilon may effectively never stop.

SVR_MAX_ITER = 200_000


def svr_grid(block_sizes=None):
    g = []
    for C in [1.0, 10.0]:
        for eps in [0.05, 0.2]:
            g.append({"kernel": "linear", "C": C, "epsilon": eps})
            g.append({"kernel": "rbf", "C": C, "epsilon": eps, "gamma": "scale"})
    return g


def svr_fit(X_tr, y_tr, X_te, hp, block_sizes=None, seed=0):
    ys, mu, sd = _standardise_y(y_tr)
    kw = {k: v for k, v in hp.items() if k != "kernel"}
    m = SVR(kernel=hp["kernel"], max_iter=SVR_MAX_ITER, cache_size=1000,
            **kw).fit(X_tr, ys)
    return m.predict(X_te) * sd + mu


# ══════════════════════════════════════════════════════════════════════
# 3. random forest
# ══════════════════════════════════════════════════════════════════════
#
# max_features is the fraction of predictors CONSIDERED at each node; the
# split itself uses the single best of them, redrawn at every node.

def rf_grid(block_sizes=None):
    return [{"max_features": mf, "min_samples_leaf": leaf}
            for mf in [0.1, 0.3, "sqrt"] for leaf in [1, 5]]


def rf_fit(X_tr, y_tr, X_te, hp, block_sizes=None, seed=0):
    m = RandomForestRegressor(n_estimators=500, n_jobs=-1,
                              random_state=seed, **hp).fit(X_tr, y_tr)
    return m.predict(X_te)


# ══════════════════════════════════════════════════════════════════════
# 4. LightGBM
# ══════════════════════════════════════════════════════════════════════

def lgbm_grid(block_sizes=None):
    return [{"num_leaves": nl, "learning_rate": lr, "feature_fraction": ff}
            for nl in [15, 31] for lr in [0.01, 0.05] for ff in [0.1, 0.5]]


def lgbm_fit(X_tr, y_tr, X_te, hp, block_sizes=None, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=1000, verbose=-1,
                          random_state=seed, **hp).fit(X_tr, y_tr)
    return m.predict(X_te)


# ══════════════════════════════════════════════════════════════════════
# 5. MLP  and  6. AGNET   (torch)
# ══════════════════════════════════════════════════════════════════════

def _torch():
    return import_module("torch")


def _train_torch(model, X_tr, y_tr, X_va, y_va, lr, wd, epochs=400,
                 batch_size=32, patience=25, min_delta=1e-4, seed=0,
                 param_groups=None):
    """Adam with early stopping on a held-out slice of the training fold."""
    torch = _torch()
    torch.manual_seed(seed)
    opt = torch.optim.Adam(param_groups if param_groups else model.parameters(),
                           lr=lr, weight_decay=wd)
    lossf = torch.nn.MSELoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_va, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.float32)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
                                     batch_size=batch_size, shuffle=True,
                                     drop_last=len(Xt) > batch_size)
    best, wait, best_state = np.inf, 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            lossf(model(xb).squeeze(-1), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(lossf(model(Xv).squeeze(-1), yv))
        if v < best - min_delta:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_torch(model, X):
    torch = _torch()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).squeeze(-1).numpy()


def mlp_grid(block_sizes=None):
    return [{"hidden": h, "dropout": d, "wd": w, "lr": 1e-3}
            for h in [(64, 32), (32, 16)]
            for d in [0.1, 0.3] for w in [1e-4, 1e-2]]


def _build_mlp(p, hidden, dropout):
    """Feed-forward network used as the MLP model and as AGNET's branch.

    Architecture, for hidden = (h1, h2):

        input   (p predictors)
          |
        Linear(p  -> h1)  ->  ReLU  ->  Dropout(rate)
          |
        Linear(h1 -> h2)  ->  ReLU  ->  Dropout(rate)
          |
        Linear(h2 -> 1)          output, one value per hybrid

    Layer sizes searched in the paper: (64, 32) and (32, 16).
    Dropout rate searched: 0.1 and 0.3.
    There are no skip connections and no batch normalisation; ReLU is the
    only nonlinearity, and dropout is active during training only.

    The list `hidden` sets both the depth and the width, so passing a
    3-tuple gives three hidden layers without any other change.
    """
    nn = _torch().nn
    layers, prev = [], p
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, 1)]                 # output layer, linear
    return nn.Sequential(*layers)


def describe_mlp(p, hidden=(64, 32), dropout=0.1):
    """Print the layer-by-layer structure and parameter count."""
    net = _build_mlp(p, hidden, dropout)
    print(f"MLP  (input {p} predictors, hidden {hidden}, dropout {dropout})")
    for i, layer in enumerate(net):
        n = sum(q.numel() for q in layer.parameters())
        print(f"  [{i}] {str(layer):45s} params {n:,}")
    total = sum(q.numel() for q in net.parameters())
    print(f"  total trainable parameters: {total:,}")
    return net


def mlp_fit(X_tr, y_tr, X_te, hp, block_sizes=None, seed=0):
    ys, mu, sd = _standardise_y(y_tr)
    n_va = max(8, int(0.15 * len(ys)))
    perm = np.random.RandomState(seed).permutation(len(ys))
    va, fit = perm[:n_va], perm[n_va:]
    model = _build_mlp(X_tr.shape[1], hp["hidden"], hp["dropout"])
    net = _train_torch(model, X_tr[fit], ys[fit], X_tr[va], ys[va],
                       lr=hp["lr"], wd=hp["wd"], seed=seed)
    return _predict_torch(net, X_te) * sd + mu


# ---------------------------------------------------------------- AGNET

class AGNET:
    """Anchored Genomic NETwork.

        yhat = mu + sum_b Z_b beta_b + tau * f(Z)

    The summation reproduces multi-matrix GBLUP: one linear path per
    relationship matrix, each with its own weight decay, so each matrix
    keeps a separate effective variance ratio. f is a feed-forward network
    on the full predictor vector.

    tau is a single learned scalar, parameterised as tau = exp(theta) with
    theta unconstrained on the real line. This enforces tau > 0 but imposes
    NO upper bound: tau is not mapped to [0, 1]. As tau -> 0 the model
    reduces exactly to multi-matrix GBLUP. theta is estimated jointly with
    all other parameters against the same objective, not tuned by
    cross-validation.

    Architecture, for predictor blocks [b1, b2, b3] and hidden = (h1, h2):

        input Z  (b1 + b2 + b3 predictors, one block per matrix)
          |
          +-- LINEAR PATHS (this is GBLUP) ------------------------+
          |     Linear(b1 -> 1, bias=True)    block 1 (G)          |
          |     Linear(b2 -> 1, bias=False)   block 2 (D)          |
          |     Linear(b3 -> 1, bias=False)   block 3 (G#G)        |
          |     -> summed                                          |
          |                                                        |
          +-- NONLINEAR BRANCH f -----------------------------+    |
                LayerNorm(b1 + b2 + b3)                       |    |
                Linear(sum -> h1) -> ReLU -> Dropout(rate)    |    |
                Linear(h1  -> h2) -> ReLU -> Dropout(rate)    |    |
                Linear(h2  ->  1)        zero-initialised     |    |
                -> scaled by tau = exp(theta)                 |    |
                                                              |    |
        output = sum(linear paths) + tau * f(Z) --------------+----+

    Only ONE intercept is fitted, carried by the first block, so the blocks
    do not compete for the mean.

    Two design choices worth noting:

    * LayerNorm is applied only inside the nonlinear branch. The linear
      paths receive the eigenvector predictors on their original scale,
      preserving the L^(1/2) weighting that makes them equivalent to GBLUP.

    * The branch output layer is initialised at zero and the linear paths at
      the ridge solution, so the model starts numerically identical to GBLUP
      and departs from it only where the validation loss improves.
    """

    def __init__(self, p, block_sizes, hidden, dropout, seed=0, theta0=0.0):
        torch = _torch()
        nn = torch.nn
        self.torch = torch
        self.blocks = block_sizes or [p]
        torch.manual_seed(seed)

        class Net(nn.Module):
            def __init__(self, blocks, hidden, dropout):
                super().__init__()
                self.blocks = blocks
                # one intercept only, carried by the first block
                self.lin = nn.ModuleList(
                    [nn.Linear(b, 1, bias=(i == 0)) for i, b in enumerate(blocks)])
                layers, prev = [nn.LayerNorm(sum(blocks))], sum(blocks)
                for h in hidden:
                    layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                    prev = h
                head = nn.Linear(prev, 1)
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                layers += [head]
                self.mlp = nn.Sequential(*layers)
                # theta; tau = exp(theta). theta0 = 0 -> tau = 1.
                self.theta = nn.Parameter(torch.tensor(float(theta0)))

            def forward(self, x):
                out, s = 0.0, 0
                for b, lin in zip(self.blocks, self.lin):
                    out = out + lin(x[:, s:s + b]).squeeze(-1)
                    s += b
                tau = torch.exp(self.theta)
                return out + tau * self.mlp(x).squeeze(-1)

        self.net = Net(self.blocks, hidden, dropout)

    def warm_start(self, X_tr, y_tr, alpha):
        """Start the linear paths at the GBLUP (ridge) solution."""
        torch = self.torch
        r = Ridge(alpha=alpha).fit(X_tr, y_tr)
        coef, s = r.coef_, 0
        with torch.no_grad():
            for b, lin in zip(self.blocks, self.net.lin):
                lin.weight.copy_(torch.from_numpy(
                    coef[s:s + b].reshape(1, -1)).float())
                if lin.bias is not None:
                    lin.bias.fill_(float(r.intercept_))
                s += b
        return self


def agnet_grid(block_sizes=None):
    nb = len(block_sizes) if block_sizes else 1
    return [{"hidden": h, "dropout": d, "wd": w, "lr": 1e-3,
             "alpha": a, "block_wd": bw}
            for h in [(64, 32), (32, 16)]
            for d in [0.1, 0.3] for w in [1e-4, 1e-2]
            for a in [10.0, 1000.0]
            for bw in ([1.0] if nb <= 1 else [1.0, 100.0])]


def agnet_fit(X_tr, y_tr, X_te, hp, block_sizes=None, seed=0):
    ys, mu, sd = _standardise_y(y_tr)
    n_va = max(8, int(0.15 * len(ys)))
    perm = np.random.RandomState(seed).permutation(len(ys))
    va, fit = perm[:n_va], perm[n_va:]

    m = AGNET(X_tr.shape[1], block_sizes, hp["hidden"], hp["dropout"], seed=seed)
    m.warm_start(X_tr[fit], ys[fit], hp["alpha"])

    # weight decay per linear path: block 0 is the reference, later blocks
    # are scaled by block_wd so an uninformative matrix can be shrunk away
    # without dragging the additive one with it
    wd, bwd = hp["wd"], hp.get("block_wd", 1.0)
    groups = []
    for i, lin in enumerate(m.net.lin):
        groups.append({"params": list(lin.parameters()),
                       "weight_decay": wd if i == 0 else wd * bwd})
    groups.append({"params": list(m.net.mlp.parameters()) + [m.net.theta],
                   "weight_decay": wd})

    net = _train_torch(m.net, X_tr[fit], ys[fit], X_tr[va], ys[va],
                       lr=hp["lr"], wd=wd, seed=seed, param_groups=groups)
    return _predict_torch(net, X_te) * sd + mu


# ══════════════════════════════════════════════════════════════════════
# registry
# ══════════════════════════════════════════════════════════════════════

MODELS = {
    "GBLUP":    (ridge_grid, ridge_fit),
    "SVR":      (svr_grid,   svr_fit),
    "RF":       (rf_grid,    rf_fit),
    "LightGBM": (lgbm_grid,  lgbm_fit),
    "MLP":      (mlp_grid,   mlp_fit),
    "AGNET":    (agnet_grid, agnet_fit),
}


# ══════════════════════════════════════════════════════════════════════
# metrics and nested cross-validation
# ══════════════════════════════════════════════════════════════════════

def predictive_ability(y, yhat):
    """Pearson correlation between observed and predicted."""
    if np.std(y) < 1e-12 or np.std(yhat) < 1e-12:
        return np.nan
    return float(np.corrcoef(y, yhat)[0, 1])


def nrmse(y, yhat):
    """RMSE normalised by the phenotypic SD. ddof=1 matches R's sd()."""
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / np.std(y, ddof=1))


def top10_overlap(y, yhat, frac=0.10):
    """Proportion of the truly best `frac` of individuals that are also
    predicted to be in the best `frac`."""
    k = max(1, int(round(frac * len(y))))
    return len(set(np.argsort(-y)[:k]) & set(np.argsort(-yhat)[:k])) / k


def prep_fold(X, y_tr, tr, te, scale_ok, k_markers=2000):
    """Feature preparation for one split. Everything phenotype-dependent
    happens on the training rows only."""
    X_tr, X_te = X[tr], X[te]
    if scale_ok and X.shape[1] > k_markers:          # marker selection
        sel = spearcor_topk(X_tr, y_tr, k_markers)
        X_tr, X_te = X_tr[:, sel], X_te[:, sel]
    return apply_scaling(X_tr, X_te, scale_ok)


def nested_cv(X, y, model_name, scale_ok, block_sizes,
              n_outer=5, n_inner=4, n_cycles=5, k_markers=2000, seed=0):
    """Outer folds estimate performance; inner folds choose hyperparameters.

    The same partitions are used for every model, so differences between
    models are not confounded with differences between folds.
    """
    grid_fn, fit_fn = MODELS[model_name]
    grid = grid_fn(block_sizes)
    rows = []

    for cycle in range(n_cycles):
        kf = KFold(n_splits=n_outer, shuffle=True, random_state=seed + cycle)
        for fold, (tr, te) in enumerate(kf.split(X), start=1):
            y_tr, y_te = y[tr], y[te]

            # ---- inner loop: choose hyperparameters ----------------
            best_hp, best_score = grid[0], -np.inf
            if len(grid) > 1:
                inner = list(KFold(n_splits=n_inner, shuffle=True,
                                   random_state=seed + cycle).split(tr))
                prepped = [(prep_fold(X, y_tr[itr], tr[itr], tr[iva],
                                      scale_ok, k_markers),
                            y_tr[itr], y_tr[iva])
                           for itr, iva in inner]
                for hp in grid:
                    scores = []
                    for (Xi_tr, Xi_va), yi_tr, yi_va in prepped:
                        try:
                            pred = fit_fn(Xi_tr, yi_tr, Xi_va, hp,
                                          block_sizes=block_sizes, seed=seed)
                            scores.append(predictive_ability(yi_va, pred))
                        except Exception:
                            scores.append(np.nan)
                    s = np.nanmean(scores) if not np.all(np.isnan(scores)) else -np.inf
                    if s > best_score:
                        best_hp, best_score = hp, s

            # ---- outer fold: refit and predict ---------------------
            X_tr, X_te = prep_fold(X, y_tr, tr, te, scale_ok, k_markers)
            pred = fit_fn(X_tr, y_tr, X_te, best_hp,
                          block_sizes=block_sizes, seed=seed)
            rows.append({"cycle": cycle + 1, "fold": fold,
                         "PA": predictive_ability(y_te, pred),
                         "NRMSE": nrmse(y_te, pred),
                         "Top10": top10_overlap(y_te, pred),
                         "best_hp": best_hp})
    return rows


# ══════════════════════════════════════════════════════════════════════
# demo
# ══════════════════════════════════════════════════════════════════════

def _demo_data(n=250, p=800, seed=0):
    """Small hybrid population with an additive trait plus a little epistasis."""
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.2, 0.8, p)
    par = 2 * (rng.random((50, p)) < freq).astype(float)
    crosses = [(i, j) for i in range(25) for j in range(25, 50)][:n]
    M = np.array([(par[i] + par[j]) / 2 for i, j in crosses])
    q = M.mean(0) / 2
    keep = (q > 0.05) & (q < 0.95)
    M = M[:, keep]
    b = rng.normal(0, 1, M.shape[1]) * (rng.random(M.shape[1]) < 0.05)
    g = M @ b
    g = g + 0.3 * (M[:, 0] * M[:, 1])              # a touch of epistasis
    g = (g - g.mean()) / g.std()
    return M, g + rng.normal(0, 0.7, len(g))


def main():
    import warnings
    warnings.filterwarnings("ignore")

    # relationship matrices (duplicated from script 01 to keep files runnable)
    M, y = _demo_data()
    X_add = M - 2 * (M.mean(0) / 2)
    p_ = M.mean(0) / 2
    A = (X_add @ X_add.T) / (2 * np.sum(p_ * (1 - p_)))
    vals, vecs = np.linalg.eigh(A)
    keep = vals[::-1] > 1e-10
    Z = (vecs[:, ::-1][:, keep]) * np.sqrt(vals[::-1][keep])

    print(f"demo population: {M.shape[0]} hybrids, {M.shape[1]} markers")
    print(f"additive eigenvector predictors: {Z.shape}\n")

    # show the network structure explicitly (torch only)
    try:
        import torch  # noqa: F401
        describe_mlp(Z.shape[1], hidden=(64, 32), dropout=0.1)
        m = AGNET(Z.shape[1], [Z.shape[1]], (64, 32), 0.1, seed=0)
        tau0 = float(torch.exp(m.net.theta))
        n_lin = sum(q.numel() for q in m.net.lin.parameters())
        n_mlp = sum(q.numel() for q in m.net.mlp.parameters())
        print(f"\nAGNET  linear paths {n_lin:,} params | branch {n_mlp:,} params "
              f"| tau at init = {tau0:.4f}")
        print("  branch:", " -> ".join(type(l).__name__ for l in m.net.mlp))
    except ImportError:
        print("(install torch to print the network structure)")
    print()

    available = []
    for name in MODELS:
        if name == "LightGBM":
            try:
                import lightgbm  # noqa: F401
            except ImportError:
                print(f"  {name:9s} skipped (lightgbm not installed)")
                continue
        if name in ("MLP", "AGNET"):
            try:
                import torch  # noqa: F401
            except ImportError:
                print(f"  {name:9s} skipped (torch not installed)")
                continue
        available.append(name)

    print(f"{'model':10s} {'PA':>7s} {'NRMSE':>8s} {'top10%':>8s}")
    print("-" * 36)
    for name in available:
        rows = nested_cv(Z, y, name, scale_ok=False, block_sizes=[Z.shape[1]],
                         n_cycles=1, seed=0)          # 1 cycle for speed
        pa = np.nanmean([r["PA"] for r in rows])
        nr = np.nanmean([r["NRMSE"] for r in rows])
        t10 = np.nanmean([r["Top10"] for r in rows])
        print(f"{name:10s} {pa:7.3f} {nr:8.3f} {t10:8.3f}")

    print("\n(5 outer folds x 1 repetition on simulated data; the paper uses "
          "5 x 5)")


if __name__ == "__main__":
    main()
