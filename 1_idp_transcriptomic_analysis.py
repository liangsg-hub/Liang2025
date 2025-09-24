# coding: utf-8
"""
Multi-K pipeline (main/auxiliary results & ORA-only version; with theme subsets)
Spearman (all genes) → Top-K → spin (spatial rotation permutation) → stability (random ROI dropout)
→ Multi-K summary/consensus → ORA (hypergeometric only; main result=consensus; auxiliary=union and each K), positive/negative split.

Output directory: OUT_DIR/spearman_topk_spin
Dependencies: pandas, numpy, statsmodels, scipy, neuromaps, matplotlib
"""

import pathlib, pickle, math, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # No display backend

# ==== Illustrator friendly: PDF vector + Type42 fonts ====
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = ['Arial']
matplotlib.rcParams['font.sans-serif'] = matplotlib.rcParams['font.family']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from scipy.stats import rankdata, norm, hypergeom
from statsmodels.stats.multitest import multipletests
from neuromaps import nulls

# ------------------- User input (modify as needed) -------------------
EXPR_PATH   = 'dk308_lh_expr_10027.csv'   # 152×N (rows=ROI)
TVAL_PATH   = 'ttest_ct_z_ha_la_results.xlsx'
TVAL_COL    = 'T'
ATLAS_PATH  = '500.aparc_lh_relabel.nii.gz'
OUT_DIR     = 'brain_gene'

# Top-K grid
K_LIST      = [100, 200, 300]
TOPK_MAX    = max(K_LIST)

# Spearman / spin settings
N_PERM      = 10000    # Number of rotations
STAT_ABS    = True     # True: |rho| two-sided; False: one-sided (by observed direction)
SEED        = 1234

# Stability evaluation
STAB_N      = 1000
STAB_DROP   = 0.10
SIGN_THR    = 0.75

# Multi-K "pass" rule (within each K)
Q_SPIN_THR  = 0.15     # FDR only within Top-K family

# ORA settings (local GMT + background)
GOBP_GMT        = 'enrichment_ana/c5.go.bp.v2023.2.Hs.symbols.gmt'
BACKGROUND_FILE = 'enrichment_ana/expr_genes.txt'
MIN_GS_SIZE     = 5
MAX_GS_SIZE     = 1000

# --------------------------------------------------------

# ------------------- Output directories -------------------
root_dir  = pathlib.Path(OUT_DIR) / 'spearman_topk_spin'
plots_dir = root_dir / 'plots'
root_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

# ------------------- Utility functions -------------------
def zrank(v):
    r = rankdata(v, method='average')     # 1..n
    r = (r - r.mean()) / r.std(ddof=0)
    return r.astype(np.float64)

def spearman_all(X_zr, y_zr):
    n = len(y_zr)
    rho = (X_zr.T @ y_zr) / n
    return np.clip(rho, -1, 1)

def nominal_p_from_rho(rho, n):
    t_stat = rho * np.sqrt((n-2) / np.maximum(1e-12, 1 - rho**2))
    p_nom  = 2 * norm.sf(np.abs(t_stat))
    return p_nom

def spin_p_for_gene(zx, yperm_zr, y_zr, stat_abs=True):
    n = len(zx)
    rho_obs  = float((zx @ y_zr) / n)
    rho_null = (zx[:, None] * yperm_zr).mean(axis=0)   # (n_perm,)
    if stat_abs:
        p = (np.sum(np.abs(rho_null) >= abs(rho_obs)) + 1) / (len(rho_null) + 1)
    else:
        if rho_obs >= 0:
            p = (np.sum(rho_null >= rho_obs) + 1) / (len(rho_null) + 1)
        else:
            p = (np.sum(rho_null <= rho_obs) + 1) / (len(rho_null) + 1)
    return rho_obs, float(p)

def stab_metrics_for_gene(vec_x, vec_y, n_rep=500, drop=0.10, seed=42):
    rng_loc = np.random.default_rng(seed)
    n = len(vec_x)
    rx = rankdata(vec_x); ry = rankdata(vec_y)
    obs_rho = float(np.corrcoef(rx, ry)[0,1])
    sign0 = np.sign(obs_rho) if obs_rho != 0 else 0.0
    keep_n = max(4, int(round(n * (1 - drop))))
    idx_all = np.arange(n)
    r_list = []
    for _ in range(n_rep):
        keep = np.sort(rng_loc.choice(idx_all, size=keep_n, replace=False))
        r = float(np.corrcoef(rankdata(vec_x[keep]), rankdata(vec_y[keep]))[0,1])
        r_list.append(r)
    r_arr = np.array(r_list)
    sign_consistency = float(np.mean(np.sign(r_arr) == sign0)) if sign0!=0 else float(np.mean(np.sign(r_arr)==0))
    return {
        'stab_sign_rate': sign_consistency,
        'stab_rho_median': float(np.nanmedian(r_arr)),
        'stab_rho_p05': float(np.nanpercentile(r_arr, 5)),
        'stab_rho_p95': float(np.nanpercentile(r_arr,95))
    }

def jaccard(a, b):
    a, b = set(a), set(b)
    u = a | b
    return len(a & b) / (len(u) if len(u)>0 else 1)

def norm_text(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower())

# ---- ORA utilities ----
def load_background(bg_path, expr_cols):
    """Return uppercase background set intersected with expression matrix columns"""
    try:
        bg = pd.read_csv(bg_path, header=None, sep=None, engine='python').iloc[:,0].astype(str).str.upper()
        bg = set(bg)
    except Exception:
        bg = set(map(str.upper, expr_cols))
    return bg & set(map(str.upper, expr_cols))

def load_gmt(gmt_path, bg_set, min_size=10, max_size=2000):
    """Read GMT → term: genes (intersected with background and filtered by size)"""
    gs = {}
    with open(gmt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            term = parts[0]
            genes = {g.upper() for g in parts[2:] if g} & bg_set
            if min_size <= len(genes) <= max_size:
                gs[term] = genes
    return gs

def ora_hypergeom(hits, gs_map, bg_set):
    """Hypergeometric ORA: return term,k,K,n,N,p,q,overlap_genes"""
    hitsU = {g.upper() for g in hits} & bg_set
    n = len(hitsU); N = len(bg_set)
    if n == 0 or N == 0 or len(gs_map) == 0:
        return pd.DataFrame(columns=['term','k','K','n','N','p','q','overlap_genes'])
    rows = []
    for term, genes in gs_map.items():
        K = len(genes)
        if K == 0:
            continue
        k = len(hitsU & genes)
        if k == 0:
            continue
        p = hypergeom.sf(k-1, N, K, n)
        rows.append((term, k, K, n, N, float(p), ",".join(sorted(hitsU & genes))))
    df = pd.DataFrame(rows, columns=['term','k','K','n','N','p','overlap_genes'])
    if not df.empty:
        df['q'] = multipletests(df['p'].values, method='fdr_bh')[1]
        df = df.sort_values(['q','p','k'], ascending=[True, True, False])
    else:
        df['q'] = []
    return df

def split_by_sign(genes, rho_map):
    pos, neg = [], []
    for g in genes:
        r = rho_map.get(g, np.nan)
        if pd.isna(r):
            continue
        (pos if r>0 else neg if r<0 else pos).append(g)  # ρ=0 goes to positive
    return pos, neg

# ------------------- Read data -------------------
expr = pd.read_csv(EXPR_PATH, index_col=0)  # (152, ~10000)
expr = expr.fillna(expr.mean())
print('Expression matrix shape:', expr.shape)

t_vec = pd.read_excel(TVAL_PATH)[TVAL_COL].values[:expr.shape[0]]
assert len(t_vec) == expr.shape[0] == 152, "t/MSN length must match expression matrix rows (152)"
t_z = (t_vec - t_vec.mean()) / t_vec.std()
print('t values/MSN standardized, length:', len(t_z))

# ------------------- Spearman (all genes) -------------------
y_zr = zrank(t_vec)
X_zr = expr.apply(zrank, axis=0).values       # (n_roi, n_gene)

rho_all = spearman_all(X_zr, y_zr)            # (n_gene,)
p_nom   = nominal_p_from_rho(rho_all, len(y_zr))
q_nom   = multipletests(p_nom, method='fdr_bh')[1]

spearman_rank = pd.DataFrame({
    'gene': expr.columns.to_numpy(),
    'rho': rho_all,
    'p_nominal': p_nom,
    'q_nominal': q_nom
}).sort_values('rho', key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)

spearman_rank.to_csv(root_dir / 'spearman_all_genes.csv', index=False)
spearman_rank[['gene','rho']].to_csv(root_dir / 'spearman_all_genes.rnk', sep='\t', header=False, index=False)
print('Exported: spearman_all_genes.csv / .rnk')

rho_map = dict(zip(spearman_rank['gene'].astype(str), spearman_rank['rho'].astype(float)))

# ------------------- Generate/load y rotations (spin nulls) -------------------
rot_path = root_dir / f'spin_y_moran_{N_PERM}.pkl'
if rot_path.exists():
    with open(rot_path,'rb') as f: rot_full = pickle.load(f)
    print('Loaded global rotations:', getattr(rot_full, 'shape', None))
else:
    rot_full = nulls.moran(
        t_z, atlas='MNI152', density='2mm', parcellation=str(ATLAS_PATH),
        n_perm=N_PERM, seed=SEED
    )
    if rot_full.shape[0] != len(t_z) and rot_full.shape[1] == len(t_z):
        rot_full = rot_full.T
    assert rot_full.shape == (len(t_z), N_PERM), f'rot_full shape error: {rot_full.shape}'
    with open(rot_path,'wb') as f: pickle.dump(rot_full, f)
    print('Saved global rotations to:', rot_path.name)

# Precompute rank-transformed permuted y (for speed)
Yperm_zr = np.apply_along_axis(zrank, 0, rot_full)  # (n_roi, n_perm)

# ------------------- Calculate spin & stability on TopK_MAX -------------------
top_all = spearman_rank.head(TOPK_MAX).copy()
spin_rho, spin_p = [], []
for g in top_all['gene']:
    zx = X_zr[:, expr.columns.get_loc(g)]
    r, p = spin_p_for_gene(zx, Yperm_zr, y_zr, stat_abs=STAT_ABS)
    spin_rho.append(r); spin_p.append(p)
top_all['rho_recalc'] = spin_rho
top_all['p_spin']     = spin_p
top_all.to_csv(root_dir / f'top{TOPK_MAX}_spin_raw.csv', index=False)

# Stability (random ROI dropout)
stab_rows = []
for g in top_all['gene']:
    x = expr[g].values.astype(float)
    m = stab_metrics_for_gene(x, t_vec, n_rep=STAB_N, drop=STAB_DROP, seed=SEED + (hash(g)%200000))
    m['gene'] = g
    stab_rows.append(m)
stab_df = pd.DataFrame(stab_rows)
stab_df.to_csv(root_dir / f'top{TOPK_MAX}_stability.csv', index=False)

# Merge main table (to TOPK_MAX)
master = top_all.merge(stab_df, on='gene', how='left')
master.to_csv(root_dir / f'top{TOPK_MAX}_spin_with_stability.csv', index=False)

# ------------------- Generate sub-results and pass lists for each K -------------------
pass_sets = {}   # Significant + stable gene set for each K
perK_tables = {}

for K in K_LIST:
    sub = master.head(K).copy()
    sub['q_spin'] = multipletests(sub['p_spin'], method='fdr_bh')[1]  # FDR only within Top-K
    sub['pass_flag'] = (sub['q_spin'] <= Q_SPIN_THR) & (sub['stab_sign_rate'] >= SIGN_THR)
    sub = sub.sort_values(['pass_flag','p_spin','stab_sign_rate','rho_recalc'], ascending=[False,True,False,False])
    sub.to_csv(root_dir / f'top{K}_spin_with_stability.csv', index=False)
    perK_tables[K] = sub
    pass_sets[K] = set(sub.loc[sub['pass_flag'], 'gene'])
    print(f'[K={K}] Passed: {len(pass_sets[K])}')

# ------------------- Multi-K: overlap & consensus (PDF plot) -------------------
# Jaccard matrix (pcolormesh → vector)
Ks = K_LIST
J = np.zeros((len(Ks), len(Ks)), dtype=float)
for i, ki in enumerate(Ks):
    for j, kj in enumerate(Ks):
        J[i,j] = jaccard(pass_sets[ki], pass_sets[kj])

fig, ax = plt.subplots(figsize=(4+0.6*len(Ks), 4+0.6*len(Ks)))
pc = ax.pcolormesh(np.arange(len(Ks)+1), np.arange(len(Ks)+1), J,
                   vmin=0, vmax=1, cmap='viridis', shading='flat')
for i in range(len(Ks)):
    for j in range(len(Ks)):
        ax.text(j+0.5, i+0.5, f'{J[i,j]:.2f}', ha='center', va='center',
                color='white' if J[i,j]>0.5 else 'black', fontsize=10)
ax.set_xticks(np.arange(len(Ks))+0.5); ax.set_xticklabels([f'K={k}' for k in Ks], rotation=45, ha='right')
ax.set_yticks(np.arange(len(Ks))+0.5); ax.set_yticklabels([f'K={k}' for k in Ks])
ax.set_xlim(0, len(Ks)); ax.set_ylim(0, len(Ks))
ax.invert_yaxis()
ax.set_title('Jaccard overlap of pass sets')
cbar = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('Jaccard')
fig.tight_layout()
plt.savefig(plots_dir / 'jaccard_pass_sets.pdf')
plt.close(fig)

# Consensus: passed in ≥2 Ks (main result set)
all_pass_rows = []
for K in K_LIST:
    df = perK_tables[K].loc[perK_tables[K]['pass_flag'], ['gene','p_spin','stab_sign_rate']].copy()
    df['K'] = K
    all_pass_rows.append(df)
all_pass = pd.concat(all_pass_rows, ignore_index=True) if all_pass_rows else pd.DataFrame(columns=['gene','K'])
cons = (all_pass.groupby('gene')
        .agg(n_K=('K','nunique'),
             mean_q=('p_spin','mean'),
             mean_stab=('stab_sign_rate','mean'))
        .reset_index())
cons2 = cons[cons['n_K']>=2].sort_values(['n_K','mean_q','mean_stab'], ascending=[False,True,False])
cons2.to_csv(root_dir / 'consensus_genes_across_K.csv', index=False)
print(f'Consensus genes (≥2 Ks stable & significant): {cons2.shape[0]}')

# Consensus heatmap (top 50 genes × K membership; PDF)
showN = min(50, cons2.shape[0])
if showN > 0:
    top_cons = cons2.head(showN)['gene'].tolist()
    M = np.zeros((showN, len(K_LIST)), dtype=int)
    for i,g in enumerate(top_cons):
        for j,K in enumerate(K_LIST):
            M[i,j] = 1 if g in pass_sets[K] else 0
    fig, ax = plt.subplots(figsize=(1.6*len(K_LIST)+4, 0.38*showN+2))
    pc2 = ax.pcolormesh(np.arange(len(K_LIST)+1), np.arange(showN+1), M,
                        cmap='Greys', vmin=0, vmax=1, shading='flat')
    ax.set_yticks(np.arange(showN)+0.5); ax.set_yticklabels(top_cons, fontsize=8)
    ax.set_xticks(np.arange(len(K_LIST))+0.5); ax.set_xticklabels([f'K={k}' for k in K_LIST])
    ax.set_xlim(0, len(K_LIST)); ax.set_ylim(0, showN)
    ax.invert_yaxis()
    ax.set_title('Consensus membership (1=pass in K)')
    plt.tight_layout()
    plt.savefig(plots_dir / 'consensus_membership_top50.pdf')
    plt.close(fig)

# Volcano plot (each K; PDF)
def volcano_plot(df, K, out_pdf):
    x = df['rho_recalc'].values
    y = -np.log10(np.clip(df['p_spin'].values, 1e-300, 1))
    pass_mask = df['pass_flag'].values
    fig, ax = plt.subplots(figsize=(6.2,4.6))
    ax.scatter(x[~pass_mask], y[~pass_mask], s=12, alpha=0.7, label='not pass')
    ax.scatter(x[pass_mask],  y[pass_mask],  s=16, alpha=0.95, label='pass')
    ax.axhline(-math.log10(0.05), ls='--', lw=1, color='grey')
    ax.set_xlabel('Spearman rho (recalc on ranks)')
    ax.set_ylabel('-log10(p_spin)')
    ax.set_title(f'Volcano (Top-{K})')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close(fig)

for K in K_LIST:
    volcano_plot(perK_tables[K], K, plots_dir / f'volcano_top{K}.pdf')

# ================== ORA: main & auxiliary results, positive/negative split ==================
# Background & full GO-BP library
if pathlib.Path(GOBP_GMT).exists():
    bg_set = load_background(BACKGROUND_FILE, expr.columns)
    pd.Series(sorted(bg_set)).to_csv(root_dir / 'enrich_background_used.txt', index=False, header=False)
    print(f'[GO] Background gene count: {len(bg_set)}')

    gs_map = load_gmt(GOBP_GMT, bg_set, MIN_GS_SIZE, MAX_GS_SIZE)
    print(f'[GO] GO-BP gene set count: {len(gs_map)}')

    # --- Main result: consensus (≥2 Ks) ---
    consensus_genes = cons2['gene'].tolist()
    cons_pos, cons_neg = split_by_sign(consensus_genes, rho_map)
    pd.Series(cons_pos).to_csv(root_dir / 'final_hits_consensus_pos.txt', index=False, header=False)
    pd.Series(cons_neg).to_csv(root_dir / 'final_hits_consensus_neg.txt', index=False, header=False)
    print(f'[Consensus] pos={len(cons_pos)}  neg={len(cons_neg)}')

    if len(cons_pos) >= 5:
        df_full_pos = ora_hypergeom(cons_pos, gs_map, bg_set)
        df_full_pos.to_csv(root_dir / f'GO_BP_ORA_consensus_pos.csv', index=False)
        n25 = int((df_full_pos['q'] < 0.25).sum()) if not df_full_pos.empty else 0
        print(f'[GO|consensus_pos] Exported {len(df_full_pos)} rows (q<0.25: {n25})')
    else:
        print(f'[GO|consensus_pos] hits={len(cons_pos)} → skipped (<5)')

    if len(cons_neg) >= 5:
        df_full_neg = ora_hypergeom(cons_neg, gs_map, bg_set)
        df_full_neg.to_csv(root_dir / f'GO_BP_ORA_consensus_neg.csv', index=False)
        n25 = int((df_full_neg['q'] < 0.25).sum()) if not df_full_neg.empty else 0
        print(f'[GO|consensus_neg] Exported {len(df_full_neg)} rows (q<0.25: {n25})')
    else:
        print(f'[GO|consensus_neg] hits={len(cons_neg)} → skipped (<5)')

    # --- Auxiliary result: union (all Ks passed union) ---
    consensus_union = set().union(*(pass_sets.get(k, set()) for k in K_LIST))
    union_pos, union_neg = split_by_sign(sorted(consensus_union), rho_map)
    pd.Series(union_pos).to_csv(root_dir / 'final_hits_union_pos.txt', index=False, header=False)
    pd.Series(union_neg).to_csv(root_dir / 'final_hits_union_neg.txt', index=False, header=False)
    print(f'[Union] pos={len(union_pos)}  neg={len(union_neg)}')

    if len(union_pos) >= 5:
        df_union_pos = ora_hypergeom(union_pos, gs_map, bg_set)
        df_union_pos.to_csv(root_dir / f'GO_BP_ORA_union_pos.csv', index=False)
        n25 = int((df_union_pos['q'] < 0.25).sum()) if not df_union_pos.empty else 0
        print(f'[GO|union_pos] Exported {len(df_union_pos)} rows (q<0.25: {n25})')
    else:
        print(f'[GO|union_pos] hits={len(union_pos)} → skipped (<5)')

    if len(union_neg) >= 5:
        df_union_neg = ora_hypergeom(union_neg, gs_map, bg_set)
        df_union_neg.to_csv(root_dir / f'GO_BP_ORA_union_neg.csv', index=False)
        n25 = int((df_union_neg['q'] < 0.25).sum()) if not df_union_neg.empty else 0
        print(f'[GO|union_neg] Exported {len(df_union_neg)} rows (q<0.25: {n25})')
    else:
        print(f'[GO|union_neg] hits={len(union_neg)} → skipped (<5)')

    # --- Auxiliary result: each K's pass set (positive/negative split) ---
    for K in K_LIST:
        genesK = list(pass_sets[K])
        posK, negK = split_by_sign(genesK, rho_map)
        pd.Series(posK).to_csv(root_dir / f'final_hits_top{K}_pos.txt', index=False, header=False)
        pd.Series(negK).to_csv(root_dir / f'final_hits_top{K}_neg.txt', index=False, header=False)
        print(f'[Top{K}] pos={len(posK)}  neg={len(negK)}')
        if len(posK) >= 5:
            df_posK = ora_hypergeom(posK, gs_map, bg_set)
            df_posK.to_csv(root_dir / f'GO_BP_ORA_top{K}_pos.csv', index=False)
            n25 = int((df_posK['q'] < 0.25).sum()) if not df_posK.empty else 0
            print(f'[GO|top{K}_pos] Exported {len(df_posK)} rows (q<0.25: {n25})')
        else:
            print(f'[GO|top{K}_pos] hits={len(posK)} → skipped (<5)')
        if len(negK) >= 5:
            df_negK = ora_hypergeom(negK, gs_map, bg_set)
            df_negK.to_csv(root_dir / f'GO_BP_ORA_top{K}_neg.csv', index=False)
            n25 = int((df_negK['q'] < 0.25).sum()) if not df_negK.empty else 0
            print(f'[GO|top{K}_neg] Exported {len(df_negK)} rows (q<0.25: {n25})')
        else:
            print(f'[GO|top{K}_neg] hits={len(negK)} → skipped (<5)')
else:
    print('[GO] GMT not found, skipped ORA (main/auxiliary results)')

print('\nDone: main/auxiliary results (positive/negative split) ORA and PDF plots')
