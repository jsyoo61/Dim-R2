# %%
import itertools as it

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils as U
import tools as T
from sklearn.metrics import r2_score as r2_score_sklearn
from tools.sklearn.metrics import r2_score

# %%
N = 100
# N = 250
C = 100
t, y_true, y_pred = U.make_signals(N, C)
y_true_og, y_pred_og = y_true.copy(), y_pred.copy()
print(y_true.shape)

# %%
var_targets = [0.01, 0.1, 0.5, 1.0]
p_noise_channels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
n_repeat = 100
l_results = []

for (var1, var2), p_noise in tqdm(it.product(it.product(var_targets, repeat=2), p_noise_channels), total=(len(var_targets)**2)*len(p_noise_channels)):
    n_noise_c = int(C*p_noise)
    for i in range(n_repeat):
        y_true, y_pred = y_true_og.copy(), y_pred_og.copy()
        y_true[:,:n_noise_c], y_pred[:,:n_noise_c] = np.random.randn(*y_true[:,:n_noise_c].shape), np.random.randn(*y_pred[:,:n_noise_c].shape)
        y_true_var, y_pred_var = np.var(y_true[:,:n_noise_c], axis=0), np.var(y_pred[:,:n_noise_c], axis=0)
        y_true[:,:n_noise_c], y_pred[:,:n_noise_c] = y_true[:,:n_noise_c]*np.sqrt(var1/y_true_var), y_pred[:,:n_noise_c]*np.sqrt(var2/y_pred_var)
        y_true_var, y_pred_var = np.var(y_true[:,:n_noise_c], axis=0), np.var(y_pred[:,:n_noise_c], axis=0)

        assert np.allclose(y_true_var, var1), f"y_true var: {y_true_var}, expected: {var1}"
        assert np.allclose(y_pred_var, var2), f"y_pred var: {y_pred_var}, expected: {var2}"
        
        old_r2 = r2_score_sklearn(y_true, y_pred, multioutput='uniform_average')
        old_r2_var = r2_score_sklearn(y_true=y_true, y_pred=y_pred, multioutput='variance_weighted')
        dim_r2 = r2_score(y_true, y_pred)
        r2_per_channel = r2_score(y_true, y_pred, axis=0)

        l_results.append({
            'var1': var1,
            'var2': var2,
            'p_noise': p_noise,
            'old_r2': old_r2,
            'old_r2_var': old_r2_var,
            'dim_r2': dim_r2,
            'r2_per_channel': r2_per_channel,
        })

df_results = pd.DataFrame(l_results)

# %%
path_save = T.Path('.')
df_agg = df_results.groupby(['var1', 'var2', 'p_noise']).agg({'old_r2': ['mean', 'std'], 'old_r2_var':['mean','std'], 'dim_r2': ['mean', 'std']}).reset_index()
# df_agg = df_results.groupby(['var1', 'var2', 'Ratio of Noise Channels']).agg({'Mean R2': ['mean', 'std'], 'Dim-R2': ['mean', 'std']}).reset_index()
df_agg
writer = pd.ExcelWriter(path_save/'r2_score_results.xlsx', engine='openpyxl')
df_results.to_excel(writer, sheet_name='all')
df_agg.to_excel(writer, sheet_name='agg')
writer.close()

# %%
exp_cols = ['var1', 'var2', 'p_noise']
gb = df_results[exp_cols+['old_r2', 'dim_r2']].groupby(by=['var1','var2','p_noise'])
mean, std = gb.mean(), gb.std()

# %%
# mean.loc[(slice(None),var1),:]
var1=0.01
# var2=0.01
var2=1.0

# %%
mean_, std_ = mean.loc[(var1, var2,slice(None)),:], std.loc[(var1, var2,slice(None)),:]
mean_sk_r2, std_sk_r2 = mean_['old_r2'].to_numpy(), std_['old_r2'].to_numpy()
mean_dim_r2, std_dim_r2 = mean_['dim_r2'].to_numpy(), std_['dim_r2'].to_numpy()

fig, ax = plt.subplots(figsize=(10,5))
x = mean_.index.get_level_values('p_noise').values
# ax.plot(x, mean_sk_r2, 'o-', label='Mean R2')
ax.errorbar(x, mean_sk_r2, fmt='o-', yerr=std_sk_r2, capsize=10, label='Mean R2')
ax.set_xlabel('Ratio of Noise Neurons')
ax.set_ylabel('Score')
ax.set_ylim(None,1)
ax.grid(True)
# ax.set_title(f'Real variance: {var1}, Predicted variance: {var2}')
ax.legend(loc='lower left')
fig.savefig(path_save/f'resilience_lineplot_meanR2_{var1}_{var2}.png', dpi=300)

ax.errorbar(x, mean_dim_r2, fmt='o-', yerr=std_dim_r2, capsize=10, label='Dim-R2')
ax.legend(loc='lower left')
fig.savefig(path_save/f'resilience_lineplot_meanR2_dimR2{var1}_{var2}.png', dpi=300)

# %%
from tableone import tableone
import scipy.stats as stats

exp_cols = ['var1', 'var2', 'p_noise']
l_sr_htest = []
for i, df_group in tqdm(df_results.groupby(exp_cols), ):
    # stat1, p1 = stats.shapiro(df_group['old_r2'])
    # stat2, p2 = stats.shapiro(df_group['dim_r2'])
    # gaussian1, gaussian2 = p1 > 0.05, p2 > 0.05
    df_score = df_group[['old_r2', 'dim_r2']].melt(var_name='R2 Type', value_name='R2 Score')
    # t1 = tableone(df_score, normal_test=True, groupby='R2 Type', pval=True, continuous=['R2 Score'], htest={'R2 Score': lambda x,y: stats.ttest_ind(x,y)[1]}, htest_name=True)
    t1 = tableone(df_score, normal_test=True, groupby='R2 Type', pval=True, continuous=['R2 Score'], htest_name=True)
    
    sr_htest = t1.htest_table.loc['R2 Score']
    sr_htest = pd.concat([sr_htest, df_group[exp_cols].iloc[0]])
    l_sr_htest.append(sr_htest)

df_htest = pd.DataFrame(l_sr_htest)

# %%
print(df_htest.nonnormal.any())
print(df_htest['Test'].unique())

# %%
df_htest = df_htest.rename(columns={'Test': 'Test Name', 'P-Value': 'p-value'})
df_htest.to_excel(path_save/'r2_score_results_htest.xlsx', index=False)

# %%
import seaborn as sns
round_n = 2
htest=False

var1_list = df_agg['var1'].unique()

rename_cols = {
    'var1': 'y_true var',
    'var2': '$y_{pred}$ noise channel variance',
    'p_noise': 'Ratio of Noise Channels',
    'old_r2': 'Mean R2',
    'old_r2_var': 'Mean R2 (Variance Weighted)',
    'dim_r2': 'Dim-R2',
}
# r2_name_list = ['old_r2', 'old_r2_var', 'dim_r2']
r2_name_list = ['old_r2', 'dim_r2']
r2_name_list = {rename_cols[k]: k for k in r2_name_list}


for var1 in var1_list:
    df_var1 = df_agg[df_agg['var1'] == var1]
    df_var1 = df_var1.rename(columns=rename_cols)
    df_var1 = df_var1.pivot(index='Ratio of Noise Channels',columns='$y_{pred}$ noise channel variance')

    # df_htest_var1 = df_htest[df_htest['var1'] == var1]
    

    fig, axes = plt.subplots(nrows=1, ncols=len(r2_name_list), figsize=(14, 3))
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    vmin, vmax = 0, 1
    for i, (r2_name, ax) in enumerate(zip(r2_name_list, axes)):
        df_var1[r2_name] = df_var1[r2_name].astype(float)
        mean_r2 = df_var1[r2_name]['mean'].round(round_n)
        std_r2 = df_var1[r2_name]['std'].round(round_n)
        df_format = mean_r2.map(lambda x: f"{x:.2f}") + '±' + std_r2.map(lambda x: f"{x:.2f}")
        # if htest:
        #     df_format += (df_htest_var1<0.05).map(lambda x: '*' if x else '')

        im = sns.heatmap(mean_r2, ax=ax, annot=df_format.values, fmt='', vmin=vmin, vmax=vmax, cmap='Blues', cbar=False)
        # im = sns.heatmap(mean_r2, ax=ax, annot=df_format.values, fmt='', vmin=vmin, vmax=vmax, cmap='Blues', )
        ax.set_title(r2_name)
    cb = fig.colorbar(im.collections[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cb.outline.set_visible(False)
    fig.suptitle(f"$y_{{true}}$ noise channel variance: {var1}", y=1.11)
    fig.patch.set_alpha(0.0)
    fig.savefig(path_save/f"r2_comparison_{var1}.png", dpi=300, bbox_inches='tight')

# %%