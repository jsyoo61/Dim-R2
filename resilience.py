# %%
import itertools as it

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

import utils as U
import tools as T
import sklearn.metrics as sk_metrics

# %%
path_save = T.Path('resilience')
path_save.mkdir(exist_ok=True)

# %%
'''
Example data diagram
'''

N = 100
C = 5
t, y_true, y_pred = U.make_signals(N, C)
y_true_og, y_pred_og = y_true.copy(), y_pred.copy()
ylim = np.abs(y_true).max() * 1.09

print(y_true.shape)

var_targets = [0.01, 0.1, 1.0]
d_noise_add = {1: 'linear_increasing', 2: 'linear_decreasing', 3: 'constant'}
l_noise = [3,4]
d_colors = {'regular': 'tab:blue', 'noise': 'grey'}
# i=[0,1]

for var1, var2 in it.product(var_targets, repeat=2):
    y_true, y_pred = y_true_og.copy(), y_pred_og.copy()

    for C_i in l_noise:
        y_true[:,C_i], y_pred[:,C_i] = np.random.randn(*y_true[:,C_i].shape), np.random.randn(*y_pred[:,C_i].shape)
    # y_true[:,i], y_pred[:,i] = np.random.randn(*y_true[:,i].shape)*noise1, np.random.randn(*y_pred[:,i].shape)*noise2
    # y_true_var, y_pred_var = np.var(y_true[:,C_i_noise], axis=0), np.var(y_pred[:,C_i_noise], axis=0)
    y_true_var, y_pred_var = np.var(y_true[:,l_noise], axis=0), np.var(y_pred[:,l_noise], axis=0)
    y_true[:,l_noise], y_pred[:,l_noise] = y_true[:,l_noise]*np.sqrt(var1/y_true_var), y_pred[:,l_noise]*np.sqrt(var2/y_pred_var)

    old_r2 = sk_metrics.r2_score(y_true, y_pred, multioutput='uniform_average')
    dim_r2 = U.r2_score(y_true, y_pred)
    r2_per_channel1 = U.r2_score(y_true, y_pred, axis=0)
    r2_per_channel2 = sk_metrics.r2_score(y_true, y_pred, multioutput='raw_values')
    assert np.allclose(r2_per_channel1, r2_per_channel2), f"r2_score: {r2_per_channel1}, r2_score_sklearn: {r2_per_channel2}"

    fig, axes = U.plot_individual(t=t, y_true=y_true, y_pred=y_pred, old_r2=old_r2, dim_r2=dim_r2, r2_per_channel=r2_per_channel1, ylim=ylim)
    # fig, axes = plot(t=t, y_true=y_true, y_pred=y_pred, old_r2=old_r2, dim_r2=dim_r2, r2_per_channel=r2_per_channel1, ylim=ylim)

    for i, ax in enumerate(axes.T[0]):
        ylabel = 'Noise' if i in l_noise else 'Neuron'
        ylabel += '\n' + rf'$R$:{r2_per_channel1[i]:.2f}'
        ax.set_ylabel(ylabel, fontsize=12)
    
    for i, axes_ in enumerate(axes):
        if i in l_noise:
            [ax.get_lines()[0].set_color(d_colors['noise']) for ax in axes_]
        else:
            [ax.get_lines()[0].set_color(d_colors['regular']) for ax in axes_]
        [ax.legend(loc='upper right') for ax in axes_]

    # axes[0,0].set_xlabel(f'$y_{{true}}$ (Noise Neuron var={var1})\nTime')
    # axes[0,1].set_xlabel(f'$y_{{pred}}$ (Noise Neuron var={var2})\nTime')
    axes[0,0].set_xlabel(f'$y$\nTime')
    axes[0,1].set_xlabel(f'$\hat{{y}}$\nTime')

    axes[0,0].xaxis.set_label_position('top')
    axes[0,1].xaxis.set_label_position('top')
    axes[-1,0].set_xticks([])  # Hide x-ticks for the last row
    axes[-1,1].set_xticks([])  # Hide x-ticks for the last row
    # axes[-1,0].xaxis.set_visible(False)
    # axes[-1,1].xaxis.set_visible(False)

    fig.suptitle(rf"Mean R2: {old_r2:.2f} / Dim-R2: {dim_r2:.2f}", y=0.95)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(path_save/f"r2_score_{var1}_{var2}.png", dpi=300, bbox_inches='tight')
    # plt.close(fig)

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
        
        old_r2 = sk_metrics.r2_score(y_true=y_true, y_pred=y_pred, multioutput='uniform_average')
        old_r2_var = sk_metrics.r2_score(y_true=y_true, y_pred=y_pred, multioutput='variance_weighted')
        dim_r2 = U.r2_score(y_true=y_true, y_pred=y_pred)
        dim_r2_axisref = U.r2_score(y_true=y_true, y_pred=y_pred, axis_norm=0, axis_pool=(0,1))
        r2_per_channel = U.r2_score(y_true=y_true, y_pred=y_pred, axis=0)

        exp_var = sk_metrics.explained_variance_score(y_true=y_true, y_pred=y_pred, multioutput='uniform_average')
        exp_var_var = sk_metrics.explained_variance_score(y_true=y_true, y_pred=y_pred, multioutput='variance_weighted')
        
        d2_abs = sk_metrics.d2_absolute_error_score(y_true=y_true, y_pred=y_pred, multioutput='uniform_average')
        # d2_abs_var = sk_metrics.d2_absolute_error_score(y_true=y_true, y_pred=y_pred, multioutput='variance_weighted')

        corr = U.pearson_corrcoef(y_true, y_pred)
        corr_per_channel = U.pearson_corrcoef(y_true, y_pred, multioutput='raw_values')

        l_results.append({
            'var1': var1,
            'var2': var2,
            'p_noise': p_noise,
            'old_r2': old_r2,
            'old_r2_var': old_r2_var,
            'dim_r2': dim_r2,
            'dim_r2_axisref': dim_r2_axisref,
            'r2_per_channel': r2_per_channel,
            'explained_var': exp_var,
            'explained_var_var': exp_var_var,
            'd2_absolute_error': d2_abs,
            # 'd2_absolute_error_var': d2_abs_var,
            'correlation': corr,
            'correlation_per_channel': corr_per_channel
        })

df_results = pd.DataFrame(l_results)

# %%
metric_cols = ['old_r2', 'dim_r2', 'dim_r2_axisref', 'old_r2_var', 'explained_var', 'explained_var_var', 'd2_absolute_error', 'correlation']
df_agg = df_results.groupby(['var1', 'var2', 'p_noise']).agg({metric: ['mean', 'std'] for metric in metric_cols}).reset_index()
df_agg

# %%
writer = pd.ExcelWriter(path_save/'r2_score_results.xlsx', engine='openpyxl')
df_results.to_excel(writer, sheet_name='all')
df_agg.to_excel(writer, sheet_name='agg')
writer.close()

# %%
'''
Lineplot (Equivalent to one column of the heatmap)
'''
exp_cols = ['var1', 'var2', 'p_noise']

gb = df_results[exp_cols+metric_cols].groupby(by=exp_cols)
mean, std = gb.mean(), gb.std()

# %%
var1=0.01
var2=1.0

# %%
mpl.rcParams['font.size']=15

mean_, std_ = mean.loc[(var1, var2,slice(None)),:], std.loc[(var1, var2,slice(None)),:]
mean_sk_r2, std_sk_r2 = mean_['old_r2'].to_numpy(), std_['old_r2'].to_numpy()
mean_dim_r2, std_dim_r2 = mean_['dim_r2'].to_numpy(), std_['dim_r2'].to_numpy()

fig, ax = plt.subplots(figsize=(5,5))
x = mean_.index.get_level_values('p_noise').values
# ax.plot(x, mean_sk_r2, 'o-', label='Mean R2')
ax.errorbar(x, mean_sk_r2, fmt='o-', yerr=std_sk_r2, capsize=10, label='Mean R2 (std)')
ax.set_xlabel('Ratio of Noise Neurons')
ax.set_ylabel('Score')
ax.set_title('Simulation with noise neurons')
# ax.set_ylim(None,1)
ax.set_ylim(None,5)
ax.grid(True)
# ax.set_title(f'Real variance: {var1}, Predicted variance: {var2}')
# ax.legend(loc='lower left')
# fig.savefig(path_save/f'resilience_lineplot_meanR2_{var1}_{var2}.png', dpi=300)

ax.errorbar(x, mean_dim_r2, fmt='o-', yerr=std_dim_r2, capsize=10, label='Dim-R2 (std)')
ax.legend(loc='lower left')

# %%
fig.savefig(path_save/f'resilience_lineplot_meanR2_dimR2{var1}_{var2}.png', dpi=300, bbox_inches='tight')

# %%
'''
optional h-test
'''
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
'''
Heatmap
'''
import seaborn as sns
import numpy as np
round_n = 2
htest=False

var1_list = df_agg['var1'].unique()

rename_cols = {
    'var1': 'y_true var',
    'var2': '$\hat{y}$ noise channel variance',
    'p_noise': 'Ratio of Noise Channels',
    'old_r2': 'Mean R2',
    'old_r2_var': 'Mean R2 (Variance Weighted)',
    'dim_r2': 'Dim-R2',
    'dim_r2_axisref': 'Dim-R2 $A_{{ref}}=1$',
    'explained_var': 'Mean EV',
    'explained_var_var': 'Mean EV (Variance Weighted)',
    'd2_absolute_error': 'Mean D2 Absolute Error',
    'correlation': 'Mean Correlation'
}
# r2_name_list = ['old_r2', 'old_r2_var', 'dim_r2']
r2_name_list = ['old_r2', 'dim_r2']
r2_name_list = {rename_cols[k]: k for k in r2_name_list}

# %%
for var1 in var1_list:
    df_var1 = df_agg[df_agg['var1'] == var1]
    df_var1 = df_var1.rename(columns=rename_cols)
    df_var1 = df_var1.pivot(index='Ratio of Noise Channels',columns='$\hat{y}$ noise channel variance')

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
    fig.suptitle(f"$y$ noise channel variance: {var1}", y=1.11)
    fig.patch.set_alpha(0.0)
    fig.savefig(path_save/f"r2_comparison_{var1}.png", dpi=300, bbox_inches='tight')

# %%
'''
Individual heatmaps per metric
'''
d_vminmax = {
    'Pearson Correlation Coefficient': (-1,1)
}

for var1 in var1_list:
    df_var1 = df_agg[df_agg['var1'] == var1]
    df_var1 = df_var1.rename(columns=rename_cols)
    df_var1 = df_var1.pivot(index='Ratio of Noise Channels',columns='$\hat{y}$ noise channel variance')

    for metric in metric_cols:
        metric_newname = rename_cols[metric]

        vmin, vmax = d_vminmax.get(metric_newname, (0,1)) # (0,1) by default, only correlation is (-1,1)

        fig, ax = plt.subplots(figsize=(7, 3))
        df_var1[metric_newname] = df_var1[metric_newname].astype(float)
        mean_r2 = df_var1[metric_newname]['mean'].round(round_n)
        std_r2 = df_var1[metric_newname]['std'].round(round_n)
        df_format = mean_r2.map(lambda x: f"{x:.2f}") + '±' + std_r2.map(lambda x: f"{x:.2f}")

        im = sns.heatmap(mean_r2, ax=ax, annot=df_format.values, fmt='', vmin=vmin, vmax=vmax, cmap='Blues', cbar=False)
        ax.set_title(metric_newname)

        cb = fig.colorbar(im.collections[0], ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cb.outline.set_visible(False)
        fig.suptitle(f"$y$ noise channel variance: {var1}", y=1.11)
        fig.patch.set_alpha(0.0)
        fig.savefig(path_save/f"{metric}_{var1}.png", dpi=300, bbox_inches='tight')

# %%
'''
Explained variance, Mean-R2, Dim-R2 for y_true vs y_true + bias
'''
bias = 3
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

        y_pred += bias

        assert np.allclose(y_true_var, var1), f"y_true var: {y_true_var}, expected: {var1}"
        assert np.allclose(y_pred_var, var2), f"y_pred var: {y_pred_var}, expected: {var2}"
        
        old_r2 = sk_metrics.r2_score(y_true=y_true, y_pred=y_pred, multioutput='uniform_average')
        old_r2_var = sk_metrics.r2_score(y_true=y_true, y_pred=y_pred, multioutput='variance_weighted')
        dim_r2 = U.r2_score(y_true=y_true, y_pred=y_pred)
        exp_var = sk_metrics.explained_variance_score(y_true=y_true, y_pred=y_pred, multioutput='uniform_average')
        exp_var_var = sk_metrics.explained_variance_score(y_true=y_true, y_pred=y_pred, multioutput='variance_weighted')
        corr = U.pearson_corrcoef(y_true, y_pred)

        l_results.append({
            'var1': var1,
            'var2': var2,
            'p_noise': p_noise,
            'old_r2': old_r2,
            'old_r2_var': old_r2_var,
            'dim_r2': dim_r2,
            'explained_var': exp_var,
            'explained_var_var': exp_var_var,
            'correlation': corr,
        })

df_results_bias = pd.DataFrame(l_results)

# %%
'''
Heatmap for above example
'''
metric_cols = ['old_r2', 'old_r2_var', 'dim_r2', 'explained_var', 'explained_var_var', 'correlation']
df_agg_bias = df_results_bias.groupby(['var1', 'var2', 'p_noise']).agg({metric: ['mean', 'std'] for metric in metric_cols}).reset_index()
var1_list = df_agg_bias['var1'].unique()

for var1 in var1_list:
    df_var1 = df_agg_bias[df_agg_bias['var1'] == var1]
    df_var1 = df_var1.rename(columns=rename_cols)
    df_var1 = df_var1.pivot(index='Ratio of Noise Channels',columns='$\hat{y}$ noise channel variance')

    for metric in metric_cols:
        metric_newname = rename_cols[metric]

        vmin, vmax = d_vminmax.get(metric_newname, (0,1)) # (0,1) by default, only correlation is (-1,1)

        fig, ax = plt.subplots(figsize=(7, 3))
        df_var1[metric_newname] = df_var1[metric_newname].astype(float)
        mean_r2 = df_var1[metric_newname]['mean'].round(round_n)
        std_r2 = df_var1[metric_newname]['std'].round(round_n)
        df_format = mean_r2.map(lambda x: f"{x:.2f}") + '±' + std_r2.map(lambda x: f"{x:.2f}")

        im = sns.heatmap(mean_r2, ax=ax, annot=df_format.values, fmt='', vmin=vmin, vmax=vmax, cmap='Blues', cbar=False)
        ax.set_title(metric_newname)

        cb = fig.colorbar(im.collections[0], ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cb.outline.set_visible(False)
        fig.suptitle(f"$y_{{true}}$ noise channel variance: {var1}", y=1.11)
        fig.patch.set_alpha(0.0)
        fig.savefig(path_save/f"biased_{metric}_{var1}.png", dpi=300, bbox_inches='tight')

# %%
