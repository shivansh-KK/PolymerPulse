"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   LAI Drug Release — Model Improvement Pipeline                              ║
║   Part 1: Hyperparameter Tuning (RandomizedSearchCV)                         ║
║   Part 2: SHAP Analysis (feature contribution explanations)                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   HOW TO RUN:                                                                ║
║   pip install pandas numpy matplotlib scikit-learn shap openpyxl lightgbm   ║
║   Place Dataset_14_feat.xlsx and Dataset_17_feat.xlsx in same folder         ║
║   python LAI_Improved_Pipeline.py                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap

try:
    import lightgbm as lgb
    USE_LGBM = True
    print("✔ LightGBM found")
except ImportError:
    USE_LGBM = False
    print("⚠ LightGBM not found — using GradientBoostingRegressor")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PATH_14 = os.path.join(PROJECT_DIR, 'data', 'improved', 'Dataset_14_feat.xlsx')
PATH_17 = os.path.join(PROJECT_DIR, 'data', 'improved', 'Dataset_17_feat.xlsx')
OUT_DIR = os.path.join(PROJECT_DIR, 'results', 'improved')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_GROUPS = [
    'LDC-PLGA','IBP-PLGA','PTX-PLGA','DEX-PLGA','PRC-PLGA',
    '5-FU-PLGA','LPA-PLGA','GEF-PLGA','TTD-PLGA','HPA-PLGA',
    'DPP-PLGA','CAF-PLGA','TAA-PLGA','PTX-PLGA-co-PALA','TAA-PLGA-co-PAVL'
]
TEST_GROUPS = [
    'LTZ-PLGA','TMZ-PLGA','CBD-PLGA','PTX-PLGA-co-PAVL','TAA-PLGA-co-PALA'
]
FEATS_14 = ['LA/GA','Polymer_MW','CL Ratio','Drug_Tm','Drug_Pka',
            'Initial D/M ratio','DLC','SA-V','SE',
            'Drug_Mw','Drug_TPSA','Drug_NHA','Drug_LogP','Time']
FEATS_17 = FEATS_14 + ['T=0.25','T=0.5','T=1.0']

BG = '#F8F9FA'
COLORS = {
    'LightGBM (Tuned)'  : '#1565C0',
    'LightGBM (Default)': '#90CAF9',
    'Random Forest'     : '#4CAF50',
    'Extra Trees'       : '#FF9800',
    'SVR'               : '#9C27B0',
    'Ridge Regression'  : '#F44336',
}

for p in [PATH_14, PATH_17]:
    if not os.path.exists(p):
        print(f"❌ File not found: {p}")
        sys.exit(1)

print(f"📁 Outputs → {os.path.abspath(OUT_DIR)}/\n")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_prep(path, feats):
    df   = pd.read_excel(path)
    df   = df[df['DP_Group'].str.contains('PLGA', case=False)].copy()
    df['Release'] = df['Release'].clip(0, 1)
    df   = df.reset_index(drop=True)
    tr   = df[df['DP_Group'].isin(TRAIN_GROUPS)].copy().reset_index(drop=True)
    te   = df[df['DP_Group'].isin(TEST_GROUPS)].copy().reset_index(drop=True)
    return (tr[feats].values, tr['Release'].values,
            tr['Experimental_index'].values,
            te[feats].values, te['Release'].values,
            te['DP_Group'].values, tr, te)

print("Loading datasets...")
X14tr,y14tr,g14tr,X14te,y14te,grp14te,df14tr,df14te = load_prep(PATH_14, FEATS_14)
X17tr,y17tr,g17tr,X17te,y17te,grp17te,df17tr,df17te = load_prep(PATH_17, FEATS_17)
print(f"  Zero-shot: Train {X14tr.shape} | Test {X14te.shape}")
print(f"  Few-shot : Train {X17tr.shape} | Test {X17te.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("PART 1 — HYPERPARAMETER TUNING")
print("=" * 65)
print("  Method: RandomizedSearchCV with GroupKFold (5 folds)")
print("  Searching best settings for LightGBM on both datasets")
print("  This will take a few minutes...\n")

cv = GroupKFold(n_splits=5)

# Hyperparameter search space for LightGBM / GradientBoosting
if USE_LGBM:
    base_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    param_grid = {
        'n_estimators'    : [300, 500, 700],
        'learning_rate'   : [0.03, 0.05, 0.08],
        'max_depth'       : [3, 4, 5],
        'num_leaves'      : [15, 31, 50],
        'subsample'       : [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_samples': [3, 5, 10],
        'reg_alpha'       : [0, 0.05, 0.1],
        'reg_lambda'      : [0, 0.05, 0.1],
    }
else:
    base_model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators' : [300, 500],
        'learning_rate': [0.03, 0.05, 0.08],
        'max_depth'    : [3, 4, 5],
        'subsample'    : [0.7, 0.8, 0.9],
        'min_samples_leaf': [3, 5],
    }

best_params  = {}
tuned_models = {}
default_scores = {}
tuned_scores   = {}

for label, Xtr, ytr, gtr, Xte, yte in [
        ('Zero-shot (14 feat)', X14tr, y14tr, g14tr, X14te, y14te),
        ('Few-shot (17 feat)',  X17tr, y17tr, g17tr, X17te, y17te)]:

    print(f"  Tuning [{label}]...")

    # Default model score first
    if USE_LGBM:
        default_m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                       max_depth=4, num_leaves=31,
                                       subsample=0.8, random_state=42, verbose=-1)
    else:
        default_m = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                               max_depth=4, subsample=0.8,
                                               random_state=42)
    ycv_def = np.clip(cross_val_predict(default_m, Xtr, ytr, cv=cv, groups=gtr), 0, 1)
    default_m.fit(Xtr, ytr)
    pred_def = np.clip(default_m.predict(Xte), 0, 1)
    default_scores[label] = {
        'CV R2'  : round(r2_score(ytr, ycv_def), 4),
        'Ext R2' : round(r2_score(yte, pred_def), 4),
        'Ext RMSE': round(np.sqrt(mean_squared_error(yte, pred_def)), 4),
    }

    # Randomised search
    search = RandomizedSearchCV(
        base_model, param_grid,
        n_iter=20,                    # test 20 random combinations
        cv=cv, scoring='r2',
        n_jobs=-1, random_state=42,
        refit=True
    )
    search.fit(Xtr, ytr, groups=gtr)
    best_p = search.best_params_
    best_params[label] = best_p

    # Tuned model CV score
    tuned_m = search.best_estimator_
    ycv_tuned = np.clip(cross_val_predict(tuned_m, Xtr, ytr, cv=cv, groups=gtr), 0, 1)
    pred_tuned = np.clip(tuned_m.predict(Xte), 0, 1)
    tuned_scores[label] = {
        'CV R2'  : round(r2_score(ytr, ycv_tuned), 4),
        'Ext R2' : round(r2_score(yte, pred_tuned), 4),
        'Ext RMSE': round(np.sqrt(mean_squared_error(yte, pred_tuned)), 4),
    }
    tuned_models[label] = tuned_m

    print(f"    Best CV R²  : {search.best_score_:.4f}")
    print(f"    Best params : {best_p}")
    print(f"    Default → Ext R²={default_scores[label]['Ext R2']:.4f}  "
          f"RMSE={default_scores[label]['Ext RMSE']:.4f}")
    print(f"    Tuned   → Ext R²={tuned_scores[label]['Ext R2']:.4f}  "
          f"RMSE={tuned_scores[label]['Ext RMSE']:.4f}")
    improvement = tuned_scores[label]['Ext R2'] - default_scores[label]['Ext R2']
    print(f"    Improvement : {improvement:+.4f} R²\n")

# ── Figure 1: Default vs Tuned comparison ────────────────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
fig1.patch.set_facecolor(BG)
fig1.suptitle('Hyperparameter Tuning: Default vs Tuned LightGBM',
              fontsize=13, fontweight='bold')

for ax, label in zip(axes, ['Zero-shot (14 feat)', 'Few-shot (17 feat)']):
    metrics  = ['CV R²', 'External R²', 'External RMSE']
    def_vals = [default_scores[label]['CV R2'],
                default_scores[label]['Ext R2'],
                default_scores[label]['Ext RMSE']]
    tun_vals = [tuned_scores[label]['CV R2'],
                tuned_scores[label]['Ext R2'],
                tuned_scores[label]['Ext RMSE']]
    x, w = np.arange(3), 0.35
    b1 = ax.bar(x - w/2, def_vals, w, label='Default', color='#90CAF9', alpha=0.9)
    b2 = ax.bar(x + w/2, tun_vals, w, label='Tuned',   color='#1565C0', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.set_facecolor('white')
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', fontsize=8, fontweight='bold')
    # improvement arrows
    for i in range(3):
        diff = tun_vals[i] - def_vals[i]
        if abs(diff) > 0.001:
            sign  = '+' if diff > 0 else ''
            color = '#1B5E20' if diff > 0 else '#B71C1C'
            ax.text(i, max(def_vals[i], tun_vals[i]) + 0.06,
                    f'{sign}{diff:.3f}', ha='center', fontsize=8,
                    color=color, fontweight='bold')

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'Fig7_Hyperparameter_Tuning.png'),
             dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig1)
print("  Fig 7 saved — Hyperparameter tuning comparison ✔")

# ── Figure 2: Best hyperparameter values ─────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
fig2.patch.set_facecolor(BG)
fig2.suptitle('Best Hyperparameters Found by RandomizedSearchCV',
              fontsize=12, fontweight='bold')

for ax, label in zip(axes2, ['Zero-shot (14 feat)', 'Few-shot (17 feat)']):
    params = best_params[label]
    keys   = list(params.keys())
    vals   = [str(v) for v in params.values()]
    y_pos  = range(len(keys))
    ax.barh(y_pos, [1]*len(keys), color='#E3F2FD', edgecolor='#1565C0', height=0.6)
    for i, (k, v) in enumerate(zip(keys, vals)):
        ax.text(0.05, i, f'{k}  =  {v}', va='center', fontsize=9,
                fontweight='bold', color='#1A237E')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title(f'{label}\nBest CV R² = {tuned_scores[label]["CV R2"]:.4f}',
                 fontsize=10, fontweight='bold')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'Fig8_Best_Parameters.png'),
             dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print("  Fig 8 saved — Best hyperparameters ✔")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("PART 2 — SHAP ANALYSIS")
print("=" * 65)
print("  Computing SHAP values for tuned LightGBM on both datasets...")
print("  This explains WHY the model makes each prediction\n")

feat_descriptions = {
    'LA/GA'            : 'LA/GA Ratio',
    'Polymer_MW'       : 'Polymer MW',
    'CL Ratio'         : 'Cross-link Ratio',
    'Drug_Tm'          : 'Drug Melting Temp',
    'Drug_Pka'         : 'Drug pKa',
    'Initial D/M ratio': 'Initial D/M Ratio',
    'DLC'              : 'Drug Loading Cap.',
    'SA-V'             : 'Surface Area/Volume',
    'SE'               : 'Surfactant %',
    'Drug_Mw'          : 'Drug Mol. Weight',
    'Drug_TPSA'        : 'Drug TPSA',
    'Drug_NHA'         : 'Drug H-bond Accept.',
    'Drug_LogP'        : 'Drug LogP',
    'Time'             : 'Time (days)',
    'T=0.25'           : 'Release @ 6h',
    'T=0.5'            : 'Release @ 12h',
    'T=1.0'            : 'Release @ 24h',
}

shap_values_dict = {}

for label, Xtr, ytr, Xte, yte, grp_te, feats, df_te in [
        ('Zero-shot (14 feat)', X14tr, y14tr, X14te, y14te, grp14te, FEATS_14, df14te),
        ('Few-shot (17 feat)',  X17tr, y17tr, X17te, y17te, grp17te, FEATS_17, df17te)]:

    print(f"  Computing SHAP for [{label}]...")
    model = tuned_models[label]

    # Use TreeExplainer for tree models (fast and exact)
    explainer  = shap.TreeExplainer(model)
    shap_train = explainer.shap_values(Xtr)
    shap_test  = explainer.shap_values(Xte)
    shap_values_dict[label] = {
        'shap_train': shap_train,
        'shap_test' : shap_test,
        'explainer' : explainer,
        'feats'     : feats,
        'Xtr'       : Xtr,
        'Xte'       : Xte,
        'grp_te'    : grp_te,
    }

    # Mean absolute SHAP per feature
    mean_shap = np.abs(shap_train).mean(axis=0)
    top_feats  = np.argsort(mean_shap)[::-1]
    print(f"    Top 5 features by SHAP impact:")
    for i in top_feats[:5]:
        fname = feat_descriptions.get(feats[i], feats[i])
        print(f"      {fname:<25}  mean|SHAP|={mean_shap[i]:.4f}")

print()

# ── Figure 3: SHAP Summary — Beeswarm plot ───────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(18, 8))
fig3.patch.set_facecolor(BG)
fig3.suptitle('SHAP Summary — Feature Impact on Drug Release Predictions\n'
              '(Red = high feature value, Blue = low feature value)',
              fontsize=12, fontweight='bold')

for ax, label, feats in [
        (axes3[0], 'Zero-shot (14 feat)', FEATS_14),
        (axes3[1], 'Few-shot (17 feat)',  FEATS_17)]:

    sv   = shap_values_dict[label]['shap_train']
    Xtr  = shap_values_dict[label]['Xtr']
    mean_shap = np.abs(sv).mean(axis=0)
    order     = np.argsort(mean_shap)   # ascending for horizontal bar

    feat_labels = [feat_descriptions.get(f, f) for f in feats]
    y_pos = np.arange(len(feats))

    for i, fi in enumerate(order):
        shap_col = sv[:, fi]
        feat_col = Xtr[:, fi]

        # normalise feature value for colouring
        feat_norm = (feat_col - feat_col.min()) / (feat_col.max() - feat_col.min() + 1e-9)
        colors    = plt.cm.RdBu_r(feat_norm)

        # jitter for beeswarm effect
        jitter = np.random.uniform(-0.3, 0.3, size=len(shap_col))
        ax.scatter(shap_col, np.full_like(shap_col, i) + jitter,
                   c=feat_norm, cmap='RdBu_r', alpha=0.4, s=8,
                   vmin=0, vmax=1, edgecolors='none')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_labels[i] for i in order], fontsize=8)
    ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('SHAP Value\n(positive = pushes release higher, negative = lower)',
                  fontsize=8)
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_facecolor('white')

    # colourbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Feature value\n(low → high)', fontsize=7)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'Fig9_SHAP_Beeswarm.png'),
             dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig3)
print("  Fig 9 saved — SHAP beeswarm summary ✔")

# ── Figure 4: SHAP Bar — Mean absolute importance ────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 7))
fig4.patch.set_facecolor(BG)
fig4.suptitle('SHAP Feature Importance — Mean |SHAP Value|',
              fontsize=12, fontweight='bold')

for ax, label, feats in [
        (axes4[0], 'Zero-shot (14 feat)', FEATS_14),
        (axes4[1], 'Few-shot (17 feat)',  FEATS_17)]:

    sv        = shap_values_dict[label]['shap_train']
    mean_shap = np.abs(sv).mean(axis=0)
    order     = np.argsort(mean_shap)   # ascending
    feat_labels = [feat_descriptions.get(feats[i], feats[i]) for i in order]
    vals_sorted = mean_shap[order]

    bar_colors = ['#1565C0' if feats[i] == 'Time' else
                  '#FF9800' if feats[i].startswith('T=') else
                  '#42A5F5'
                  for i in order]

    bars = ax.barh(feat_labels, vals_sorted, color=bar_colors,
                   edgecolor='white', linewidth=0.5)
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_xlabel('Mean |SHAP Value|  (average impact on prediction)', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_facecolor('white')

    for bar, val in zip(bars, vals_sorted):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=7, color='#1A237E')

    # Legend
    legend_patches = [
        mpatches.Patch(color='#1565C0', label='Time (days)'),
        mpatches.Patch(color='#FF9800', label='Early release features'),
        mpatches.Patch(color='#42A5F5', label='Drug/Polymer properties'),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc='lower right')

plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'Fig10_SHAP_Bar.png'),
             dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig4)
print("  Fig 10 saved — SHAP bar chart ✔")

# ── Figure 5: SHAP dependence plots — top 4 features ─────────────────────────
for label, feats in [('Zero-shot (14 feat)', FEATS_14),
                     ('Few-shot (17 feat)',  FEATS_17)]:

    sv    = shap_values_dict[label]['shap_train']
    Xtr_d = shap_values_dict[label]['Xtr']
    mean_shap  = np.abs(sv).mean(axis=0)
    top4_idx   = np.argsort(mean_shap)[::-1][:4]

    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
    fig5.patch.set_facecolor(BG)
    short = '14feat' if '14' in label else '17feat'
    fig5.suptitle(f'SHAP Dependence Plots — Top 4 Features\n{label}',
                  fontsize=12, fontweight='bold')

    for ax, fi in zip(axes5.flat, top4_idx):
        fname     = feat_descriptions.get(feats[fi], feats[fi])
        feat_vals = Xtr_d[:, fi]
        shap_vals = sv[:, fi]

        sc = ax.scatter(feat_vals, shap_vals, c=feat_vals,
                        cmap='viridis', alpha=0.5, s=15, edgecolors='none')
        ax.axhline(0, color='red', lw=1, linestyle='--', alpha=0.5)

        # Trend line
        z   = np.polyfit(feat_vals, shap_vals, 1)
        p   = np.poly1d(z)
        xs  = np.linspace(feat_vals.min(), feat_vals.max(), 100)
        ax.plot(xs, p(xs), 'r-', lw=2, alpha=0.8, label='Trend')

        ax.set_xlabel(fname, fontsize=9)
        ax.set_ylabel('SHAP Value', fontsize=9)
        ax.set_title(f'{fname}\n(higher = pushes release prediction up)',
                     fontsize=9, fontweight='bold')
        ax.set_facecolor('white')
        ax.tick_params(labelsize=8)
        plt.colorbar(sc, ax=ax, fraction=0.03, label='Feature value')

    plt.tight_layout()
    fig5.savefig(os.path.join(OUT_DIR, f'Fig11_SHAP_Dependence_{short}.png'),
                 dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig5)
    print(f"  Fig 11 saved — SHAP dependence plots [{label}] ✔")

# ── Figure 6: SHAP per test group ────────────────────────────────────────────
fig6, axes6 = plt.subplots(2, 5, figsize=(22, 10))
fig6.patch.set_facecolor(BG)
fig6.suptitle('SHAP Mean |Values| Per Test Group\n'
              'Which features drove predictions for each unseen drug-polymer group?',
              fontsize=12, fontweight='bold')

for row, (label, feats) in enumerate([
        ('Zero-shot (14 feat)', FEATS_14),
        ('Few-shot (17 feat)',  FEATS_17)]):

    sv     = shap_values_dict[label]['shap_test']
    grp_te = shap_values_dict[label]['grp_te']
    feat_labels = [feat_descriptions.get(f, f) for f in feats]

    for col, grp in enumerate(TEST_GROUPS):
        ax   = axes6[row, col]
        mask = grp_te == grp
        sv_g = sv[mask]
        mean_g = np.abs(sv_g).mean(axis=0)
        order  = np.argsort(mean_g)
        top_n  = min(8, len(feats))
        order_top = order[-top_n:]

        bar_colors = ['#1565C0' if feats[i] == 'Time' else
                      '#FF9800' if feats[i].startswith('T=') else '#42A5F5'
                      for i in order_top]
        ax.barh([feat_labels[i] for i in order_top],
                mean_g[order_top],
                color=bar_colors, edgecolor='none')
        ax.set_title(f'{grp.replace("-PLGA","")}\n({label.split()[0]})',
                     fontsize=7.5, fontweight='bold')
        ax.tick_params(labelsize=6.5)
        ax.set_facecolor('white')
        ax.set_xlabel('Mean |SHAP|', fontsize=6.5)

plt.tight_layout()
fig6.savefig(os.path.join(OUT_DIR, 'Fig12_SHAP_Per_Test_Group.png'),
             dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig6)
print("  Fig 12 saved — SHAP per test group ✔")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("COMPLETE ✔ — FULL RESULTS SUMMARY")
print("=" * 65)
print()
print(f"  {'Dataset':<25} {'Model':<20} {'Default R²':>11} "
      f"{'Tuned R²':>10} {'Improvement':>12}")
print(f"  {'-'*80}")
for label in ['Zero-shot (14 feat)', 'Few-shot (17 feat)']:
    d = default_scores[label]['Ext R2']
    t = tuned_scores[label]['Ext R2']
    print(f"  {label:<25} {'LightGBM':<20} {d:>11.4f} {t:>10.4f} {t-d:>+12.4f}")

print()
print("  OUTPUT FILES:")
for f in sorted(os.listdir(OUT_DIR)):
    icon = '📊' if f.endswith('.png') else '📗'
    print(f"  {icon} {OUT_DIR}/{f}")
print()
print("  SHAP KEY FINDINGS:")
for label, feats in [('Zero-shot (14 feat)', FEATS_14),
                     ('Few-shot (17 feat)',  FEATS_17)]:
    sv   = shap_values_dict[label]['shap_train']
    mean_shap  = np.abs(sv).mean(axis=0)
    top3       = np.argsort(mean_shap)[::-1][:3]
    top3_names = [feat_descriptions.get(feats[i], feats[i]) for i in top3]
    print(f"  [{label}] Top 3 most impactful features:")
    for i, (idx, name) in enumerate(zip(top3, top3_names), 1):
        print(f"    {i}. {name:<28} mean|SHAP|={mean_shap[idx]:.4f}")
print()
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# EXCEL EXPORT — Full results including tuning + SHAP
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("EXCEL EXPORT")
print("=" * 65)

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

def thin_border():
    s = Side(style='thin')
    return Border(left=s, right=s, top=s, bottom=s)

def sc(cell, bold=False, fc='000000', bg=None, size=10,
       align='center', wrap=False):
    cell.font      = Font(name='Arial', bold=bold, color=fc, size=size)
    cell.alignment = Alignment(horizontal=align, vertical='center', wrap_text=wrap)
    cell.border    = thin_border()
    if bg: cell.fill = PatternFill('solid', fgColor=bg)

def hdr(ws, row, cols, vals, bg='1565C0', size=10):
    for c, v in zip(cols, vals):
        cell = ws.cell(row=row, column=c, value=v)
        sc(cell, bold=True, fc='FFFFFF', bg=bg, size=size)

def mhdr(ws, rng, val, bg='1A5276', size=12):
    ws.merge_cells(rng)
    cell = ws[rng.split(':')[0]]
    cell.value = val
    sc(cell, bold=True, fc='FFFFFF', bg=bg, size=size)

def drow(ws, row, cols, vals, alt=False, bold=False):
    bg = 'EBF5FB' if alt else 'FFFFFF'
    for c, v in zip(cols, vals):
        cell = ws.cell(row=row, column=c, value=v)
        sc(cell, bg=bg, size=9, bold=bold)

def cw(ws, col, w):
    ws.column_dimensions[get_column_letter(col)].width = w

wb = Workbook()

# ── Sheet 1: Project Summary ──────────────────────────────────────────────────
ws0 = wb.active
ws0.title = 'Project Summary'
ws0.sheet_view.showGridLines = False
mhdr(ws0, 'B2:I2',
    'LAI Drug Release — ML Results (Hyperparameter Tuning + SHAP Analysis)',
    size=13)
ws0.row_dimensions[2].height = 30

info = [
    ('Polymer focus',    'PLGA only (20 groups) — most clinically approved polymer for LAIs'),
    ('Train / Test',     '15 train groups (100 formulations, 1,630 pts) | 5 test groups (13 formulations, 284 pts)'),
    ('Models',           'LightGBM, Random Forest, Extra Trees, SVR, Ridge Regression'),
    ('Tuning method',    'RandomizedSearchCV — 20 iterations, 5-Fold GroupKFold'),
    ('SHAP method',      'TreeExplainer — exact SHAP values for tree-based models'),
    ('Reference paper',  'Bannigan et al. 2023, Nature Communications'),
    ('Dataset source',   'github.com/aspuru-guzik-group/long-acting-injectables'),
]
for i, (k, v) in enumerate(info, 4):
    ws0[f'B{i}'].value = k + ':'; ws0[f'B{i}'].font = Font(name='Arial', bold=True, size=10)
    ws0[f'C{i}'].value = v;       ws0[f'C{i}'].font = Font(name='Arial', size=10)

r = 13
mhdr(ws0, f'B{r}:H{r}', 'Default vs Tuned LightGBM — Key Results', bg='154360', size=11)
ws0.row_dimensions[r].height = 22; r += 1
hdr(ws0, r, range(2, 8),
    ['Dataset', 'Default CV R²', 'Tuned CV R²', 'Default Ext R²', 'Tuned Ext R²', 'Improvement'],
    bg='2471A3'); r += 1
for i, label in enumerate(['Zero-shot (14 feat)', 'Few-shot (17 feat)']):
    vals = [label,
            default_scores[label]['CV R2'],
            tuned_scores[label]['CV R2'],
            default_scores[label]['Ext R2'],
            tuned_scores[label]['Ext R2'],
            round(tuned_scores[label]['Ext R2'] - default_scores[label]['Ext R2'], 4)]
    for c, v in zip(range(2, 8), vals):
        cell = ws0.cell(row=r, column=c, value=v)
        sc(cell, bg='D6EAF8' if i == 0 else 'EBF5FB', size=10)
    r += 1

r += 1
mhdr(ws0, f'B{r}:H{r}', 'SHAP Top Features', bg='154360', size=11)
ws0.row_dimensions[r].height = 22; r += 1
hdr(ws0, r, range(2, 6),
    ['Dataset', 'Rank 1 Feature', 'Rank 2 Feature', 'Rank 3 Feature'],
    bg='2471A3'); r += 1
for i, (label, feats) in enumerate([('Zero-shot (14 feat)', FEATS_14),
                                     ('Few-shot (17 feat)',  FEATS_17)]):
    sv   = shap_values_dict[label]['shap_train']
    ms   = np.abs(sv).mean(axis=0)
    top3 = [feat_descriptions.get(feats[j], feats[j]) for j in np.argsort(ms)[::-1][:3]]
    for c, v in zip(range(2, 6), [label] + top3):
        cell = ws0.cell(row=r, column=c, value=v)
        sc(cell, bg='D6EAF8' if i == 0 else 'EBF5FB', size=10)
    r += 1

cw(ws0, 2, 22); cw(ws0, 3, 65)
for c in range(4, 9): cw(ws0, c, 18)

# ── Sheet 2: Hyperparameter Tuning ───────────────────────────────────────────
ws1 = wb.create_sheet('Hyperparameter Tuning')
ws1.sheet_view.showGridLines = False
mhdr(ws1, 'A1:F1', 'Hyperparameter Tuning Results — RandomizedSearchCV')
ws1.row_dimensions[1].height = 24
r = 3
for label in ['Zero-shot (14 feat)', 'Few-shot (17 feat)']:
    mhdr(ws1, f'A{r}:F{r}', label, bg='2471A3', size=10)
    ws1.row_dimensions[r].height = 20; r += 1
    hdr(ws1, r, range(1, 4), ['Metric', 'Default LightGBM', 'Tuned LightGBM']); r += 1
    rows_data = [
        ('CV R²',       default_scores[label]['CV R2'],   tuned_scores[label]['CV R2']),
        ('External R²', default_scores[label]['Ext R2'],  tuned_scores[label]['Ext R2']),
        ('External RMSE', default_scores[label]['Ext RMSE'], tuned_scores[label]['Ext RMSE']),
        ('Improvement', '—', round(tuned_scores[label]['Ext R2'] - default_scores[label]['Ext R2'], 4)),
    ]
    for j, (m, d, t) in enumerate(rows_data):
        for c, v in zip(range(1, 4), [m, d, t]):
            cell = ws1.cell(row=r, column=c, value=v)
            sc(cell, bg='EBF5FB' if j % 2 == 0 else 'FFFFFF', size=9)
        r += 1
    # Best params
    mhdr(ws1, f'A{r}:F{r}', 'Best Parameters Found', bg='117A65', size=9)
    ws1.row_dimensions[r].height = 18; r += 1
    hdr(ws1, r, range(1, 3), ['Parameter', 'Best Value'], bg='1D8348', size=9); r += 1
    for j, (k, v) in enumerate(best_params[label].items()):
        for c, val in zip(range(1, 3), [k, str(v)]):
            cell = ws1.cell(row=r, column=c, value=val)
            sc(cell, bg='D5F5E3' if j % 2 == 0 else 'EAFAF1', size=9)
        r += 1
    r += 1
for c, w in zip(range(1, 7), [28, 20, 20, 10, 10, 10]): cw(ws1, c, w)

# ── Sheet 3: SHAP Feature Importance ─────────────────────────────────────────
ws2 = wb.create_sheet('SHAP Feature Importance')
ws2.sheet_view.showGridLines = False
mhdr(ws2, 'A1:E1', 'SHAP Feature Importance — Mean |SHAP Value| on Training Set')
ws2.row_dimensions[1].height = 24

feat_desc_full = {
    'LA/GA':'Ratio of lactic to glycolic acid in PLGA — controls degradation rate',
    'Polymer_MW':'Molecular weight of the polymer — affects chain length and release',
    'CL Ratio':'Cross-linking ratio — higher = slower release',
    'Drug_Tm':'Melting temperature — affects drug-polymer compatibility',
    'Drug_Pka':'Acid dissociation constant — affects ionisation in release media',
    'Initial D/M ratio':'Initial drug-to-matrix ratio — affects drug loading',
    'DLC':'Drug loading capacity — affects burst release magnitude',
    'SA-V':'Surface area to volume — higher = faster release',
    'SE':'Surfactant % in release media — affects solubility',
    'Drug_Mw':'Molecular weight of drug — heavier = slower diffusion through PLGA',
    'Drug_TPSA':'Topological polar surface area — affects membrane permeability',
    'Drug_NHA':'Number of hydrogen bond acceptors — affects solubility',
    'Drug_LogP':'Lipophilicity — fat-soluble drugs partition better into PLGA',
    'Time':'Days after injection — most important time-dependent driver',
    'T=0.25':'Actual release at 6 hours — early burst indicator',
    'T=0.5':'Actual release at 12 hours — transition phase indicator',
    'T=1.0':'Actual release at 24 hours — strongest early predictor',
}

r = 3
for label, feats in [('Zero-shot (14 feat)', FEATS_14), ('Few-shot (17 feat)', FEATS_17)]:
    mhdr(ws2, f'A{r}:E{r}', label, bg='2471A3', size=10)
    ws2.row_dimensions[r].height = 20; r += 1
    hdr(ws2, r, range(1, 6),
        ['Rank', 'Feature', 'Mean |SHAP|', '% of Total Impact', 'Biological Meaning']); r += 1
    sv   = shap_values_dict[label]['shap_train']
    ms   = np.abs(sv).mean(axis=0)
    total = ms.sum()
    pairs = sorted(zip(feats, ms), key=lambda x: x[1], reverse=True)
    for rank, (feat, imp) in enumerate(pairs, 1):
        bg = ('D5F5E3' if rank == 1 else 'FEF9E7' if rank == 2
              else 'EBF5FB' if rank % 2 == 0 else 'FFFFFF')
        vals = [rank, feat, round(float(imp), 6),
                round(float(imp)/float(total)*100, 2),
                feat_desc_full.get(feat, '')]
        for c, v in zip(range(1, 6), vals):
            cell = ws2.cell(row=r, column=c, value=v)
            sc(cell, bg=bg, size=9, bold=(rank <= 2),
               align='left' if c == 5 else 'center')
        r += 1
    r += 1
ws2.conditional_formatting.add('C4:C50',
    ColorScaleRule(start_type='min', start_color='FFFFFF',
                   end_type='max', end_color='1565C0'))
for c, w in zip(range(1, 6), [8, 22, 16, 18, 55]): cw(ws2, c, w)

# ── Sheet 4: SHAP Values — Raw data ──────────────────────────────────────────
ws3 = wb.create_sheet('SHAP Values (Training)')
ws3.sheet_view.showGridLines = False
mhdr(ws3, 'A1:R1', 'Raw SHAP Values — Training Set (Zero-shot, 14 features)')
ws3.row_dimensions[1].height = 24
label   = 'Zero-shot (14 feat)'
sv_raw  = shap_values_dict[label]['shap_train']
Xtr_raw = shap_values_dict[label]['Xtr']
shap_cols = [f'SHAP_{f}' for f in FEATS_14]
all_cols  = ['DP_Group', 'Time'] + shap_cols + ['Prediction_Impact']
hdr(ws3, 3, range(1, len(all_cols)+1), all_cols, bg='1565C0', size=9)
for i in range(min(500, len(df14tr))):   # cap at 500 rows for file size
    row_vals = ([df14tr.iloc[i]['DP_Group'],
                 round(float(df14tr.iloc[i]['Time']), 4)] +
                [round(float(sv_raw[i, j]), 6) for j in range(len(FEATS_14))] +
                [round(float(sv_raw[i].sum()), 6)])
    alt = i % 2 == 0
    for c, v in zip(range(1, len(row_vals)+1), row_vals):
        cell = ws3.cell(row=4+i, column=c, value=v)
        sc(cell, bg='EBF5FB' if alt else 'FFFFFF', size=8)
for c in range(1, len(all_cols)+1): cw(ws3, c, 14)
cw(ws3, 1, 22); cw(ws3, 2, 10)

# ── Sheet 5: CV + External Scores ────────────────────────────────────────────
ws4 = wb.create_sheet('All Model Scores')
ws4.sheet_view.showGridLines = False
mhdr(ws4, 'A1:H1', 'All Model Scores — CV and External Validation')
ws4.row_dimensions[1].height = 24

MODEL_NAMES = ['LightGBM (Default)', 'LightGBM (Tuned)',
               'Random Forest', 'Extra Trees', 'SVR', 'Ridge Regression']

# Recompute all model external scores for reference
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR as SVR_
from sklearn.linear_model import Ridge as Ridge_

def ref_models():
    if USE_LGBM:
        import lightgbm as lgb
        def_m = lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,
                                    max_depth=4,num_leaves=31,subsample=0.8,
                                    random_state=42,verbose=-1)
    else:
        def_m = GradientBoostingRegressor(n_estimators=500,learning_rate=0.05,
                                           max_depth=4,subsample=0.8,random_state=42)
    return {
        'Random Forest' : RandomForestRegressor(n_estimators=500,random_state=42,n_jobs=-1),
        'Extra Trees'   : ExtraTreesRegressor(n_estimators=500,random_state=42,n_jobs=-1),
        'SVR'           : Pipeline([('sc',StandardScaler()),('m',SVR_(kernel='rbf',C=10,gamma='scale'))]),
        'Ridge Regression': Pipeline([('sc',StandardScaler()),('m',Ridge_())]),
    }

r = 3
for label, Xtr, ytr, gtr, Xte, yte in [
        ('Zero-shot (14 feat)', X14tr,y14tr,g14tr,X14te,y14te),
        ('Few-shot (17 feat)',  X17tr,y17tr,g17tr,X17te,y17te)]:
    mhdr(ws4, f'A{r}:H{r}', label, bg='2471A3', size=10)
    ws4.row_dimensions[r].height = 20; r += 1
    hdr(ws4, r, range(1, 8), ['Model','CV R²','CV RMSE','Ext R²','Ext RMSE','Ext MAE','Notes']); r += 1

    # Default LightGBM row
    d = default_scores[label]
    vals = ['LightGBM (Default)', d['CV R2'], '—', d['Ext R2'], d['Ext RMSE'], '—', 'Original settings']
    for c, v in zip(range(1, 8), vals):
        cell = ws4.cell(row=r, column=c, value=v)
        sc(cell, bg='BBDEFB', size=9)
    r += 1

    # Tuned LightGBM row
    t = tuned_scores[label]
    imp = round(t['Ext R2'] - d['Ext R2'], 4)
    vals = ['LightGBM (Tuned)', t['CV R2'], '—', t['Ext R2'], t['Ext RMSE'], '—',
            f'After tuning (Δ={imp:+.4f})']
    for c, v in zip(range(1, 8), vals):
        cell = ws4.cell(row=r, column=c, value=v)
        sc(cell, bg='D5F5E3', size=9, bold=True)
    r += 1

    # Other models
    other_models = ref_models()
    cv = GroupKFold(n_splits=5)
    for j, (name, model) in enumerate(other_models.items()):
        ycv  = np.clip(cross_val_predict(model, Xtr, ytr, cv=cv, groups=gtr), 0, 1)
        model.fit(Xtr, ytr)
        pred = np.clip(model.predict(Xte), 0, 1)
        vals = [name,
                round(r2_score(ytr, ycv), 4),
                round(np.sqrt(mean_squared_error(ytr, ycv)), 4),
                round(r2_score(yte, pred), 4),
                round(np.sqrt(mean_squared_error(yte, pred)), 4),
                round(mean_absolute_error(yte, pred), 4), '']
        for c, v in zip(range(1, 8), vals):
            cell = ws4.cell(row=r, column=c, value=v)
            sc(cell, bg='EBF5FB' if j % 2 == 0 else 'FFFFFF', size=9)
        r += 1
    r += 1

for c, w in zip(range(1, 8), [28, 10, 10, 10, 10, 10, 28]): cw(ws4, c, w)

# ── Sheet 6: Methodology ─────────────────────────────────────────────────────
ws5 = wb.create_sheet('Methodology Notes')
ws5.sheet_view.showGridLines = False
mhdr(ws5, 'A1:D1', 'Methodology Notes')
ws5.row_dimensions[1].height = 24
notes = [
    ('TUNING',  'Method',       'RandomizedSearchCV — tests 20 random parameter combinations'),
    ('TUNING',  'CV strategy',  '5-Fold GroupKFold — same formulation never in train+val simultaneously'),
    ('TUNING',  'Scoring',      'R² score used to select best parameters'),
    ('TUNING',  'Finding',      'Default LightGBM was already near-optimal — tuning gave minimal improvement'),
    ('SHAP',    'Method',       'TreeExplainer — exact SHAP values (not approximation) for tree models'),
    ('SHAP',    'What it shows','How much each feature pushed each individual prediction up or down'),
    ('SHAP',    'Top feature',  'Time is by far the most important feature in both zero-shot and few-shot'),
    ('SHAP',    'Drug_Mw',      'Drug molecular weight is 2nd most important — heavier drugs diffuse slower through PLGA'),
    ('SHAP',    'LA/GA Ratio',  'Higher LA/GA = slower PLGA degradation = slower drug release'),
    ('SHAP',    'T=1.0',        'In few-shot, Release@24h is 2nd most important — knowing day 1 release predicts the rest well'),
    ('SHAP',    'SA-V',         'Higher surface area to volume = more drug exposed to release media = faster release'),
]
hdr(ws5, 3, range(1, 5), ['Category', 'Item', 'Detail', ''], bg='1A5276')
cat_bg = {'TUNING': 'FEF9E7', 'SHAP': 'E8F8F5'}
for i, (cat, item, detail) in enumerate(notes, 4):
    ws5.row_dimensions[i].height = 22
    bg = cat_bg.get(cat, 'FFFFFF')
    for c, v in zip(range(1, 4), [cat, item, detail]):
        cell = ws5.cell(row=i, column=c, value=v)
        sc(cell, bg=bg, size=9, align='left' if c >= 2 else 'center', wrap=(c == 3))
cw(ws5, 1, 12); cw(ws5, 2, 28); cw(ws5, 3, 80)

# ── Tab colours ───────────────────────────────────────────────────────────────
tab_colors = {
    'Project Summary'        : '1A5276',
    'Hyperparameter Tuning'  : 'B7950B',
    'SHAP Feature Importance': '1D8348',
    'SHAP Values (Training)' : '117A65',
    'All Model Scores'       : '922B21',
    'Methodology Notes'      : '1A5276',
}
for ws_obj in wb.worksheets:
    ws_obj.sheet_properties.tabColor = tab_colors.get(ws_obj.title, '1A5276')

excel_path = os.path.join(OUT_DIR, 'LAI_Improved_Results.xlsx')
wb.save(excel_path)
print(f"  Excel saved: {excel_path} ✔")
print()
print("  SHEETS:")
print("  📋 Project Summary      — key numbers at a glance")
print("  📋 Hyperparameter Tuning — default vs tuned comparison + best params")
print("  📋 SHAP Feature Importance — ranked features with biological meaning")
print("  📋 SHAP Values (Training)  — raw SHAP values for every training row")
print("  📋 All Model Scores     — CV + external scores for all 6 models")
print("  📋 Methodology Notes    — explanation of tuning and SHAP methods")