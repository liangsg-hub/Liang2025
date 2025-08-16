# coding: utf-8
"""
Workflow:
1. abagen maps AHBA microarray to left 152 ROI template.
2. Read 8D MSN or t values and perform z-score.
3. Calculate simple Spearman correlation.
4. Use neuromaps.nulls.moran to generate permutation distribution and obtain p-value.
5. Sparse PLS regression (SPLS) + bootstrap core gene selection
"""

import pathlib
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr, zscore
from neuromaps.parcellate import Parcellater
from neuromaps import nulls, stats

# ---------- User needs to modify these paths ----------
ATLAS_PATH = "/FS_SUBJECTS/fsaverageSubP/parcellation/500.aparc_lh_relabel.nii.gz"
MSN_data    = "/group_comparison/ttest_ct_z_ha_la_results.xlsx"
OUT_DIR    = "/fs_processed_files2/msn_8meas_files/brain_gene"
FUNC_IMG_PATH = "/data_analysis/shen_2mm_268_parcellation.nii.gz"
N_PERM     = 10000
SEED       = 1234
# ------------------------------------------------------

out_dir = pathlib.Path(OUT_DIR).expanduser().resolve()
out_dir.mkdir(parents=True, exist_ok=True)

# 1. abagen get expression matrix
expr_path = 'dk308_lh_expr_10027.csv'
expr = pd.read_csv(expr_path, index_col=0)

print("Expression matrix shape:", expr.shape)
expr = expr.fillna(expr.mean()) # Fill NaN values by column mean

# 2. Read t values / MSN and z-score
t_vec = pd.read_excel(MSN_data)['T'].values[:152]
if len(t_vec) != 152:
    raise ValueError("t/MSN vector length must be 152")
t_z = zscore(t_vec)

# 3. Spearman correlation
parc = Parcellater(ATLAS_PATH, "MNI152")
# Parcellate a functional annotation image and correlate
func_img = nib.load(FUNC_IMG_PATH)
func_vec = parc.fit_transform(func_img, "MNI152").ravel()
rho_emp, _ = spearmanr(t_z, func_vec)

# 4. Moran permutation
import pickle

rot_path = out_dir / "dk308_152_moran_1w.pkl"
if rot_path.exists():
    with open(rot_path, "rb") as f:
        rot = pickle.load(f)
    print("Loaded saved rot permutation result:", rot.shape)
else:
    rot = nulls.moran(
        t_z,
        atlas="MNI152",
        density="2mm",
        parcellation=ATLAS_PATH,
        n_perm=N_PERM,
        seed=SEED
    )  # (152, N_PERM)
    with open(rot_path, "wb") as f:
        pickle.dump(rot, f)
    print("Saved rot permutation result:", rot.shape)

print("rot shape:", rot.shape)
print("func_vec shape:", func_vec.shape)

# ---------- 5. Sparse PLS regression (SPLS) ----------
from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict

# --------------------------- 1. Preprocessing ---------------------------
y = t_z
B = 1000
core_thr = 0.05
rng = np.random.RandomState(42)

# 1) Low variance filter: keep top 90 percent variance genes
var = expr.var(axis=0).values
thr = np.quantile(var, 0.10)           # Remove lowest 10 percent
vt = VarianceThreshold(threshold=thr)
X_red = vt.fit_transform(expr.values)
genes_red = expr.columns[vt.get_support()]
n_roi, n_gene_red = X_red.shape
print(f"Retained {n_gene_red} genes (threshold={thr:.3g})")

# 2) Find best alpha for Lasso (run once)
X_scaled = StandardScaler().fit_transform(X_red)
alpha_best = LassoCV(
    alphas=np.logspace(-4, -1, 40),    # Do not use alpha=1, too coarse
    cv=3,
    n_jobs=-1,
    max_iter=10000,
    random_state=SEED
).fit(X_scaled, y).alpha_
print(f"Best Lasso alpha = {alpha_best:.5f}")

# 3) Build bootstrap pipeline
pipe = make_pipeline(
    StandardScaler(),                  # Put inside pipeline
    SelectFromModel(
        Lasso(alpha=alpha_best, max_iter=10000, random_state=SEED), # best Lasso alpha = 0.10000
        threshold='median', max_features=100,
    ),
    PLSRegression(n_components=1, scale=False)
)

# --------------------------- 2. Bootstrap ---------------------------
select_counts = Counter()
gene_weights  = defaultdict(list)

print(f"Start bootstrap x{B} ...")
for _ in range(B):
    idx = rng.choice(n_roi, n_roi, replace=True)
    pipe.fit(X_red[idx], y[idx])

    sel_mask   = pipe.named_steps['selectfrommodel'].get_support()
    sel_genes  = genes_red[sel_mask]            # Direct gene names
    pls_w      = pipe.named_steps['plsregression'].x_weights_[:, 0]

    select_counts.update(sel_genes)
    for g, w in zip(sel_genes, pls_w):
        gene_weights[g].append(w)

# --------------------------- 3. Results summary ---------------------------
# 3.1 Selection frequency
freq_series = (pd.Series(select_counts) / B).sort_values(ascending=False)
freq_series.to_csv(out_dir / "bootstrap_gene_frequency.csv", header=["freq"])

# Select core genes by frequency
core_genes = freq_series[freq_series >= core_thr].index.tolist()
pd.Series(core_genes).to_csv(out_dir / "core_genes_freq.csv", index=False)

# 3.2 Weight analysis
w_mean = {g: np.mean(ws) for g, ws in gene_weights.items()}
w_series = pd.Series(w_mean, name="mean_weight")
w_series.to_csv(out_dir / "bootstrap_gene_weights.csv")

# Select genes by weight (for example, top 10 percent by absolute value)
w_abs = w_series.abs().sort_values(ascending=False)
weight_top_genes = w_abs.head(int(len(w_abs) * 1)).index.tolist()
pd.Series(weight_top_genes).to_csv(out_dir / "top_weight_genes.csv", index=False)

# Combined selection: genes with both high frequency and high weight
combined_genes = [g for g in core_genes if g in weight_top_genes]
pd.Series(combined_genes).to_csv(out_dir / "high_freq_weight_genes.csv", index=False)

print(f"Top 10 by frequency:\n{freq_series.head(10)}")
print(f"Core genes ({len(core_genes)}): {core_genes[:10]} ...")
print(f"Top 10 by weight: {weight_top_genes[:10]} ...")
print(f"High frequency and high weight genes ({len(combined_genes)}): {combined_genes[:10] if combined_genes else 'None'} ...")

# ---------------------------------------------------------------------
# 4. Permutation test based on stable genes
# ---------------------------------------------------------------------
from neuromaps import nulls
import pickle

N_PERM_SPIN  = 1000
perm_cache = out_dir / f"perm_coregenes_{N_PERM_SPIN}.pkl"

# 1) Extract core gene expression matrix and standardize to z-score
core_mask  = np.isin(genes_red, combined_genes)
X_core_raw = X_red[:, core_mask]              # shape = (152, n_core)
X_core     = StandardScaler().fit_transform(X_core_raw)

pls = PLSRegression(n_components=1, scale=False)
pls.fit(X_core, y)
pls_score = pls.x_scores_[:, 0]  # First component scores
pls_var = np.var(pls_score)      # Variance of first component
print(f"Core gene PLS score variance: {pls_var:.4f}")
pd.Series(pls_score, name="pls1_score").to_csv(out_dir / "core_gene_pls1_scores.csv", index=False)

# Save PLS first component gene weights and loadings
pls_weights = pd.Series(pls.x_weights_[:, 0], index=combined_genes, name="pls1_weight")
pls_weights.to_csv(out_dir / "core_gene_pls1_weights.csv")

pls_loadings = pd.Series(pls.x_loadings_[:, 0], index=combined_genes, name="pls1_loading")
pls_loadings.to_csv(out_dir / "core_gene_pls1_loadings.csv")

# 2) Calculate empirical statistic (Spearman rho or PLS R2)
core_score = X_core.mean(axis=1) # Simple signature; can also use PLS score
rho_emp, _ = spearmanr(core_score, y)
print(f"Core gene signature vs y: rho_emp = {rho_emp:.3f}")

# 3) Generate or load null distribution
if perm_cache.exists():
    with open(perm_cache, "rb") as f:
        y_rot = pickle.load(f)
    print(f"Loaded permutation matrix y_rot: {y_rot.shape}")
else:
    y_rot = nulls.moran(
        y,
        atlas="MNI152", density="2mm",
        parcellation=ATLAS_PATH,
        n_perm=N_PERM_SPIN, seed=SEED
    )  # shape = (152, N_PERM)
    with open(perm_cache, "wb") as f:
        pickle.dump(y_rot, f)
    print(f"Saved y_rot to {perm_cache.name}")

# 4) Calculate null distribution statistics
from joblib import Parallel, delayed
rho_null = Parallel(n_jobs=-1)(
    delayed(lambda i: spearmanr(core_score, y_rot[:, i])[0])(i)
    for i in range(N_PERM_SPIN)
)
p_spin = (np.sum(np.abs(rho_null) >= abs(rho_emp)) + 1) / (N_PERM_SPIN + 1)
print(f"Permutation test p_spin = {p_spin:.4f}")

pd.DataFrame({"rho_null": rho_null}).to_csv(out_dir / "perm_rho_null.csv", index=False)
with open(out_dir / "perm_summary.txt", "w") as f:
    f.write(f"rho_emp = {rho_emp:.4f}\n")
    f.write(f"p_spin = {p_spin:.5f}\n")

# 5) PLS permutation test
rho_emp_pls, _ = spearmanr(pls_score, y)
rho_null_pls = Parallel(n_jobs=-1)(
    delayed(lambda i: spearmanr(pls_score, y_rot[:, i])[0])(i)
    for i in range(N_PERM_SPIN)
)
print(f"PLS score vs y: rho_emp = {rho_emp_pls:.3f}")
p_spin_pls = (np.sum(np.abs(rho_null_pls) >= abs(rho_emp_pls)) + 1) / (N_PERM_SPIN + 1)
print(f"PLS permutation test p_spin = {p_spin_pls:.4f}")

pd.DataFrame({"rho_null_pls": rho_null_pls}).to_csv(out_dir / "pls_perm_rho_null.csv", index=False)
with open(out_dir / "pls_perm_summary.txt", "w") as f:
    f.write(f"rho_emp_pls = {rho_emp_pls:.4f}\n")
    f.write(f"p_spin_pls = {p_spin_pls:.5f}\n")