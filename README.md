# Hybrid crop genomic prediction — worked example

Reference implementation for:

> Vitale, P., Gerard, G., Pérez-Rodríguez, P., D., Govindan, V., Crossa, J.
> *Exploring Machine- and Deep-Learning Algorithms for Hybrid Crop Prediction.*

Two self-contained scripts. Both run out of the box on simulated data, so you
can see the whole pipeline before plugging in your own genotypes.

```bash
python 01_relationship_matrices.py
python 02_models.py
```

## 01_relationship_matrices.py

Hybrid reconstruction and the predictors.

| Function | What it does |
|---|---|
| `make_hybrids` | reconstructs hybrid genotypes in silico from parental allele dosages (0/1/2) |
| `filter_maf` | drops markers below MAF 0.05 in the reconstructed hybrid matrix |
| `vanraden_A` | additive relationship matrix **G**, VanRaden (2008) method 1 |
| `vitezica_D` | dominance relationship matrix **D**, Vitezica et al. (2013) |
| `epistasis_AA` | additive-by-additive **G#G**, Hadamard square rescaled by the mean diagonal |
| `zstar` | eigenvector predictors **Z\*** = **UΛ**^½ |
| `build_features` | assembles the five predictor sets used in the paper |

## 02_models.py

The six models and the nested cross-validation.

| Model | Notes |
|---|---|
| GBLUP | ridge on **Z\***, with a separate variance ratio per relationship matrix |
| SVR | linear and RBF kernels; γ fixed at `"scale"` = 1/(*p*·Var(**X**)) |
| RF | 500 trees; `max_features` ∈ {0.1, 0.3, sqrt} |
| LightGBM | 1000 rounds, leaf-wise growth, histogram splits |
| MLP | two hidden layers, ReLU, dropout, early stopping |
| AGNET | multi-matrix GBLUP with an embedded neural branch (this paper) |

`nested_cv` runs 5 outer folds × 5 repetitions for performance, with 4 inner
folds for hyperparameter selection. The same partitions are used for every
model, so differences between models are not confounded with differences
between folds.

## Two details that matter

**Do not standardise the eigenvector columns.** The **Λ**^½ weighting *is* the
GBLUP prior — it encodes how much variance each direction carries. Ridge on
**UΛ**^½ is algebraically identical to GBLUP; scaling the columns to unit
variance destroys that equivalence. In our wheat data predictive ability fell
from 0.549 to −0.013 when this guard was removed. `apply_scaling()` enforces it.

**Keep all eigenvectors.** Every component with λ > 10⁻¹⁰ is retained, so this
is a rotation of the feature space, not a dimensionality reduction. For **G**
the columns of **Z\*** are the principal component scores of the centred marker
matrix.

## AGNET

```
ŷ = μ + Σ_b Z_b β_b + τ · f(Z)
```

The summation reproduces multi-matrix GBLUP, one linear path per relationship
matrix, each with its own weight decay. `f` is a feed-forward network on the
full predictor vector.

τ is a single learned scalar, parameterised as τ = exp(θ) with θ unconstrained
on the real line. This enforces τ > 0 but imposes **no upper bound** — τ is not
mapped to [0, 1]. As τ → 0 the model reduces exactly to multi-matrix GBLUP. θ is
estimated jointly with all other parameters against the same objective, not
tuned by cross-validation.

The linear paths are warm-started at the ridge solution and the branch output
layer is initialised at zero, so the model begins numerically identical to
GBLUP and departs from it only where the validation loss improves.

`theta0` in the `AGNET` constructor sets the initial value of θ.

## Requirements

```
numpy  scipy  scikit-learn        # required
torch                             # optional: MLP and AGNET
lightgbm                          # optional: LightGBM
```

Models whose dependency is missing are skipped with a message rather than
failing.

## Using your own data

Replace `simulate_parents()` in script 01 with a loader returning an
`(n_parents × n_markers)` array of allele dosages coded 0/1/2, plus a list of
`(parent1, parent2)` index pairs. Everything downstream is unchanged.

## References

- VanRaden, P.M. (2008) *J. Dairy Sci.* 91:4414–4423
- Vitezica, Z.G., Varona, L., Legarra, A. (2013) *Genetics* 195:1223–1230
- Nogueira, S., Sechidis, K., Brown, G. (2018) *JMLR* 18:1–54 — feature-selection stability
- Piles, M. et al. (2021) *Front. Genet.* 12:611506 — Spearman filter for marker preselection
