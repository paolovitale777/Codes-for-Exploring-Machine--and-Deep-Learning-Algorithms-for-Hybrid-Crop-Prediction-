#!/usr/bin/env python3
"""
01_relationship_matrices.py
===========================
Hybrid genotype reconstruction, genomic relationship matrices, and the
eigenvector predictors used in:

    Vitale et al. "Exploring Machine- and Deep-Learning Algorithms for
    Hybrid Crop Prediction".

What this script does
---------------------
1. reconstructs hybrid genotypes in silico from parental allele dosages
2. builds the additive (G), dominance (D) and additive-by-additive (G#G)
   genomic relationship matrices
3. turns each matrix into a set of predictors by eigendecomposition

Run it as-is to see all three steps on a small simulated population:

    python 01_relationship_matrices.py

To use your own data, replace `simulate_parents()` with a loader that returns
a (n_parents x n_markers) array of allele dosages coded 0/1/2 and a list of
(parent1, parent2) index pairs.

Requires: numpy
"""

from __future__ import annotations

import numpy as np

MAF_MIN = 0.05          # markers below this minor allele frequency are dropped
EIG_TOL = 1e-10         # eigenvalues at or below this are treated as zero


# ----------------------------------------------------------------------
# 1. hybrid genotype reconstruction
# ----------------------------------------------------------------------

def make_hybrids(M_par: np.ndarray, crosses) -> np.ndarray:
    """Reconstruct hybrid genotypes from parental allele dosages.

    Parents are coded 0/1/2 (copies of the reference allele). For a cross
    between two inbred parents:

        parent1 = 0, parent2 = 0  ->  hybrid = 0     (both homozygous, same)
        parent1 = 2, parent2 = 2  ->  hybrid = 2     (both homozygous, same)
        parent1 = 0, parent2 = 2  ->  hybrid = 1     (heterozygous)

    which is simply the average of the two parental dosages. Averaging also
    handles residual heterozygosity in a parent sensibly, so it is applied
    directly rather than as a special case.

    Parameters
    ----------
    M_par   (n_parents, n_markers) array of allele dosages
    crosses iterable of (i, j) index pairs into the rows of M_par

    Returns
    -------
    (n_hybrids, n_markers) array of hybrid allele dosages
    """
    M_par = np.asarray(M_par, dtype=np.float64)
    i = np.fromiter((a for a, _ in crosses), dtype=int)
    j = np.fromiter((b for _, b in crosses), dtype=int)
    return (M_par[i] + M_par[j]) / 2.0


def filter_maf(M: np.ndarray, maf_min: float = MAF_MIN) -> np.ndarray:
    """Drop markers whose minor allele frequency falls below `maf_min`.

    Applied to the RECONSTRUCTED HYBRID matrix, not to the parents, because
    the hybrid population is what the models are fitted to.
    """
    p = M.mean(axis=0) / 2.0
    keep = (p > maf_min) & (p < 1.0 - maf_min)
    return M[:, keep]


# ----------------------------------------------------------------------
# 2. genomic relationship matrices
# ----------------------------------------------------------------------

def vanraden_A(M: np.ndarray) -> np.ndarray:
    """Additive genomic relationship matrix, VanRaden (2008) method 1.

        G = W W' / (2 * sum p_k q_k),      W = M - 2p

    where p_k is the reference allele frequency at marker k.
    """
    X = np.asarray(M, dtype=np.float64)
    p = X.mean(axis=0) / 2.0
    W = X - 2.0 * p
    return (W @ W.T) / (2.0 * np.sum(p * (1.0 - p)))


def vitezica_D(M: np.ndarray) -> np.ndarray:
    """Dominance relationship matrix, Vitezica et al. (2013).

    The dominance design matrix contrasts the heterozygote against the two
    homozygotes, orthogonally to the additive effects:

        genotype 0 (aa)  ->  -2 q^2       [coded here from dosage 0]
        genotype 1 (Aa)  ->   2 p q
        genotype 2 (AA)  ->  -2 p^2

    Note this is NOT the Endelman & Jannink parameterisation; the two differ
    in scaling and in orthogonality to the additive matrix.
    """
    X = np.asarray(M, dtype=np.float64)
    p = X.mean(axis=0) / 2.0
    q = 1.0 - p
    Wd = np.where(X == 0, -2.0 * p ** 2,
                  np.where(X == 1, 2.0 * p * q, -2.0 * q ** 2))
    return (Wd @ Wd.T) / np.sum((2.0 * p * q) ** 2)


def epistasis_AA(A: np.ndarray) -> np.ndarray:
    """Additive-by-additive epistatic matrix as the Hadamard square of G.

        G#G = (G o G) / mean(diag(G o G))

    The Hadamard (element-wise) product of two Gram matrices is the Gram
    matrix of the element-wise products of the underlying features, so G#G
    is the relationship matrix implied by all pairwise marker products
    without ever forming them. Rescaling by the mean diagonal puts it on a
    comparable scale to G and D so that their variance components are
    interpretable side by side.
    """
    AA = A * A
    return AA / np.mean(np.diag(AA))


def build_kernels(M_hyb: np.ndarray) -> dict:
    """All three relationship matrices from a hybrid marker matrix."""
    A = vanraden_A(M_hyb)
    return {"A": A, "D": vitezica_D(M_hyb), "AA": epistasis_AA(A)}


# ----------------------------------------------------------------------
# 3. eigenvector predictors
# ----------------------------------------------------------------------

def zstar(K: np.ndarray, tol: float = EIG_TOL) -> np.ndarray:
    """Eigenvector predictors Z* = U L^(1/2) for a relationship matrix K.

    K is symmetric positive semi-definite, so K = U L U'. Setting

        Z* = U L^(1/2)      gives      Z* Z*' = K      exactly.

    Two properties make this the right representation:

    * Ridge regression on Z* IS GBLUP. Writing g = Z* a with a ~ N(0, I s2),
      Var(g) = Z* Z*' s2 = K s2, which is the GBLUP model. Nothing is lost
      and nothing is approximated.

    * Because all eigenvectors with a non-negligible eigenvalue are kept,
      this is a ROTATION of the feature space, not a dimensionality
      reduction. For G in particular, the columns of Z* are the principal
      component scores of the centred marker matrix.

    IMPORTANT: do not standardise the columns of Z*. The L^(1/2) weighting
    is the GBLUP prior -- it encodes how much variance each direction
    carries. Scaling the columns to unit variance destroys the equivalence
    with GBLUP and, in our data, collapsed predictive ability from 0.55 to
    approximately zero. See `apply_scaling` in 02_models.py.
    """
    vals, vecs = np.linalg.eigh(K)
    vals = vals[::-1]
    vecs = vecs[:, ::-1]
    keep = vals > tol
    return vecs[:, keep] * np.sqrt(vals[keep])


# the five predictor sets compared in the paper
KERNEL_SETS = {
    "A":    ("A",),                 # G
    "AD":   ("A", "D"),             # G + D
    "AAA":  ("A", "AA"),            # G + G#G
    "ADAA": ("A", "D", "AA"),       # G + D + G#G
}


def build_features(representation: str, kernels: dict, M_hyb: np.ndarray):
    """Predictors for one representation.

    Returns (X, scale_ok, block_sizes). `scale_ok` tells the modelling code
    whether the columns may be standardised: True for raw markers, False for
    eigenvector predictors (see the warning in `zstar`).
    """
    if representation == "markers":
        return M_hyb, True, [M_hyb.shape[1]]

    blocks = [zstar(kernels[k]) for k in KERNEL_SETS[representation]]
    return np.hstack(blocks), False, [b.shape[1] for b in blocks]


# ----------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------

def simulate_parents(n_par=60, n_markers=1500, seed=0):
    """A toy inbred panel: dosages in {0, 2}, as for fully inbred lines."""
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.1, 0.9, n_markers)
    return 2 * (rng.random((n_par, n_markers)) < freq).astype(np.float64)


def main():
    rng = np.random.default_rng(1)

    # --- 1. reconstruct hybrids ------------------------------------
    M_par = simulate_parents()
    n_par = M_par.shape[0]
    crosses = [(i, j) for i in range(0, 30) for j in range(30, n_par)][:300]
    M_hyb = make_hybrids(M_par, crosses)
    M_hyb = filter_maf(M_hyb)
    print(f"parents  : {M_par.shape[0]} x {M_par.shape[1]} markers")
    print(f"hybrids  : {M_hyb.shape[0]} x {M_hyb.shape[1]} markers after MAF filter")
    print(f"dosages  : {sorted(np.unique(M_hyb))[:5]} ...")

    # --- 2. relationship matrices ----------------------------------
    K = build_kernels(M_hyb)
    print("\nrelationship matrices")
    for name, mat in K.items():
        vals = np.linalg.eigvalsh(mat)
        print(f"  {name:3s} shape {mat.shape}  rank {int((vals > EIG_TOL).sum()):4d}  "
              f"mean diagonal {np.mean(np.diag(mat)):.3f}")

    # --- 3. eigenvector predictors ---------------------------------
    print("\neigenvector predictors")
    for rep in ["markers"] + list(KERNEL_SETS):
        X, scale_ok, bs = build_features(rep, K, M_hyb)
        print(f"  {rep:8s} {X.shape[0]:4d} x {X.shape[1]:5d}   blocks {bs}"
              f"   standardise columns: {scale_ok}")

    # --- the identity that motivates the whole approach ------------
    Z = zstar(K["A"])
    err = np.abs(Z @ Z.T - K["A"]).max()
    print(f"\ncheck: max |Z* Z*' - G| = {err:.2e}  (should be ~0)")


if __name__ == "__main__":
    main()
