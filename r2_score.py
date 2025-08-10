# %%
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2_score_sklearn
import itertools as it
from matplotlib import cm
from tqdm import tqdm
import pandas as pd
import matplotlib as mpl

import tools as T
import tools.sklearn.metrics
import utils as U

# %%
'''
Environment setting
'''
path_save = T.Path('dim_eval')
path_save.mkdir(parents=True, exist_ok=True)

# %%
'''
Create dataset
'''
D = 1000
N = 100
C = 5

t, y_true, y_pred = U.make_signals(N, C, stagger_phase=False)
y_true, y_pred = np.tile(y_true, (D, 1, 1)), np.tile(y_pred, (D, 1, 1)) # (trials, time, neuron)
print(y_true.shape)

noise_true = np.zeros(y_true.shape)
noise_pred = np.zeros(y_pred.shape)

# %%
'''
Add noise to the signals
'''
d_noise_add = {1: 'linear_increasing', 2: 'linear_decreasing', 3: 'constant'}

for C_i, noisetype in d_noise_add.items():
    noise1 = np.stack([U.noise(y_true_, noisetype=noisetype) for y_true_ in y_true[:,:,C_i]], axis=0)
    noise2 = np.stack([U.noise(y_pred_, noisetype=noisetype) for y_pred_ in y_pred[:,:,C_i]], axis=0)
    
    noise_true[:, :, C_i] = noise1
    noise_pred[:, :, C_i] = noise2

y_true += noise_true
y_pred += noise_pred

print(noise_true.var(axis=(0,1)))
print(noise_pred.var(axis=(0,1)))
# %%
'''
substitute with noise
'''
var1, var2 = y_true[:,:,0].var(), y_pred[:,:,0].var() # Same variance for both signals

l_noise = [4]  # Add noise to channel 4
for C_i in l_noise:
    noise1 = np.random.randn(*y_true[:, :, C_i].shape)
    noise2 = np.random.randn(*y_pred[:, :, C_i].shape)
    
    noise1 = noise1*np.sqrt(var1 / noise1.var(axis=1, keepdims=True))
    noise2 = noise2*np.sqrt(var2 / noise2.var(axis=1, keepdims=True))
    
    y_true[:, :, C_i] = noise1
    y_pred[:, :, C_i] = noise2

print(y_true.var(axis=(0,1)))
plt.plot(y_true[0])
plt.matshow(y_true[0], aspect='auto')

# %%
'''
Create similar dataset with consistent/varying neural bias
'''
bias_consistent = np.broadcast_to(np.arange(y_true.shape[-1]), y_true.shape)
bias_varying = np.stack([np.random.permutation(bias_consistent_.T).T  for bias_consistent_ in bias_consistent], axis=0)

data_dict = {
    'Time Varying': {
        'y_true': y_true,
        'y_pred': y_pred,
        
    },
    'Consistent Neural Bias': {
        'y_true': y_true + bias_consistent,
        'y_pred': y_pred + bias_consistent,
    },
    'Varying Neural Bias': {
        'y_true': y_true + bias_varying,
        'y_pred': y_pred + bias_varying,
    }
}

# Add mean prediction to each dataset
for data_name, data in data_dict.items():
    y_true, y_pred = data['y_true'], data['y_pred']    
    y_pred_mean = np.broadcast_to(y_pred.mean(axis=1, keepdims=True), y_pred.shape)
    y_pred_mean.shape
    y_pred_mean[0,:,2][0]
    y_pred[0,:,2].mean()
    help(y_pred.mean)
    y_pred_mean[0,:,2]
    y_pred.mean(axis=1, keepdims=True)[0,0,0]
    y_pred.mean(axis=1, keepdims=True)[0,:,0]
    y_pred.mean(axis=1, keepdims=True).shape

for data_name, data in data_dict.items():
    path_save_data = path_save / data_name
    path_save_data.mkdir(parents=True, exist_ok=True)

# %%
'''
Sample waveforms
'''
def plot_sample(y, xlim_sync=False):
    '''
    y: (time, neuron)
    '''
    fig, axes = plt.subplots(ncols=y.shape[1], figsize=(2*y.shape[1], 5), sharey=True)
    yticks = np.arange(y.shape[0])
    for i, (ax, y_) in enumerate(zip(axes, y.T)):
        ax.plot(y_, yticks)
        ax.invert_yaxis()
        ax.set_xlabel(f'N{i} ($\mu$={int(np.round(y_.mean())):d})')
    if xlim_sync:
        xlim = np.abs(np.array([ax.get_xlim() for ax in axes])).max()
        [ax.set_xlim(-xlim, xlim) for ax in axes]
    axes[0].set_ylabel('Time')
    return fig, axes

data_i = 1

y = data_dict['Time Varying']['y_true'][data_i]
fig, axes = plot_sample(y, xlim_sync=True)
xlim = axes[0].get_xlim()
xlim_range = xlim[1] - xlim[0]
fig.savefig(path_save/f"sample_time_varying.png", dpi=300, bbox_inches='tight')

y = data_dict['Consistent Neural Bias']['y_true'][data_i]
fig, axes = plot_sample(y, xlim_sync=False)
xlim_mean = np.round(y.mean(axis=0)).astype(int)
xlim = np.stack([xlim_mean - xlim_range/2, xlim_mean + xlim_range/2], axis=1)
[ax.set_xlim(xlim_ ) for ax, xlim_ in zip(axes, xlim)]
[ax.set_xticks([xlim_mean_ -2, xlim_mean_, xlim_mean_ + 2]) for ax, xlim_mean_ in zip(axes, xlim_mean)]
fig.savefig(path_save/f"sample_consistent_neural_bias.png", dpi=300, bbox_inches='tight')

y = data_dict['Varying Neural Bias']['y_true'][data_i]
fig, axes = plot_sample(y, xlim_sync=False)
xlim_mean = np.round(y.mean(axis=0)).astype(int)
xlim = np.stack([xlim_mean - xlim_range/2, xlim_mean + xlim_range/2], axis=1)
[ax.set_xlim(xlim_ ) for ax, xlim_ in zip(axes, xlim)]
[ax.set_xticks([xlim_mean_ -2, xlim_mean_, xlim_mean_ + 2]) for ax, xlim_mean_ in zip(axes, xlim_mean)]
fig.savefig(path_save/f"sample_varying_neural_bias.png", dpi=300, bbox_inches='tight')

# %%
'''
2D Dim-R2 score comparison
'''
axis_list = list(range(y_true.ndim))
axislabel_list = ['Data', 'Time', 'Neuron']
axislabel_dict = {i:axislabel for i, axislabel in enumerate(axislabel_list)}
axislabel_dict[None] = 'None'

figsize=(4,4)
nrows = 2*len(axis_list)
ncols = len(axis_list)
suptitle_y = 1.02
subfigtitlesize = None
hspace = 0.3
wspace = 0.4

def add_colorbar(ax):
    """
    Add a colorbar to the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Explanation variable
    """
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = ax.figure.colorbar(cm.ScalarMappable(norm=ax.images[0].norm, cmap=ax.images[0].cmap), cax=cax)
    return cbar

def plot_r2_combinations_extensive(y_true, y_pred, axis_ref=None, axis_bias=None):
    # fig = plt.figure(figsize=(figsize[0]*ncols, figsize[1]*nrows), constrained_layout=True)
    set_axis_ref = set(axis_ref) if isinstance(axis_ref, Iterable) else {axis_ref}

    fig = plt.figure(figsize=(figsize[0]*ncols, figsize[1]*nrows))
    fig.subplots_adjust(hspace=hspace)
    subfigs = fig.subfigures(nrows=nrows)

    for axis, subfig in zip(axis_list, subfigs[0:nrows//2]):
        axis_remaining = list(set(axis_list) - {axis})

        axes = subfig.subplots(ncols=ncols, nrows=1)
        score, subfig, axes = U.r2_score_plot(y_true, y_pred, axis=axis, axis_ref=axis_ref, axis_bias=axis_bias, fig=subfig, axes=axes)
        [ax.set_xlabel(axislabel_list[axis_remaining[1]]) for ax in axes[[0,2]]]
        [ax.set_ylabel(axislabel_list[axis_remaining[0]]) for ax in axes[[0,2]]]
        
        axis_ref_remaining = list(set(axis_list) - {axis} - set_axis_ref)
        if len(axis_ref_remaining) == 2:
            axes[1].set_ylabel(axislabel_list[axis_ref_remaining[0]])
            axes[1].set_xlabel(axislabel_list[axis_ref_remaining[1]])
        elif len(axis_ref_remaining) == 1:
            axes[1].set_xlabel(axislabel_list[axis_ref_remaining[0]])

        axis_name = axislabel_list[axis]
        # axis_ref_name = axislabel_list[axis_ref] if axis_ref is not None else 'None'
        # axis_bias_name = axislabel_list[axis_bias] if axis_bias is not None else 'None'

        subfig.suptitle(f"Axis: ({axis_name})", y=suptitle_y, fontsize=subfigtitlesize)

    # 2D (1D result)
    for (i, j), subfig in zip(it.combinations(axis_list, 2), subfigs[nrows//2:]):
        axis_remaining = list(set(axis_list) - {i, j})

        axes = subfig.subplots(ncols=ncols, nrows=1)
        score, subfig, axes = U.r2_score_plot(y_true, y_pred, axis=(i,j), axis_ref=axis_ref, axis_bias=axis_bias, fig=subfig, axes=axes)
        [ax.set_xlabel(axislabel_list[axis_remaining[0]]) for ax in axes[[0,2]]]

        axis_ref_remaining = list(set(axis_list) - {i,j} - set_axis_ref)
        if len(axis_ref_remaining) == 2:
            axes[1].set_ylabel(axislabel_list[axis_ref_remaining[0]])
            axes[1].set_xlabel(axislabel_list[axis_ref_remaining[1]])
        elif len(axis_ref_remaining) == 1:
            axes[1].set_xlabel(axislabel_list[axis_ref_remaining[0]])

        axis_names = axislabel_list[i],axislabel_list[j]
        # axis_ref_name = axislabel_list[axis_ref] if axis_ref is not None else 'None'
        # axis_bias_name = axislabel_list[axis_bias] if axis_bias is not None else 'None'

        subfig.suptitle(f"Axis: ({axis_names[0]},{axis_names[1]})", y=suptitle_y, fontsize=subfigtitlesize)

    [add_colorbar(ax) for ax in axes[:,-1]]
    
    return fig, axes

def plot_r2_combinations(y_true, y_pred, axis_ref_bias_list):
    fig = plt.figure(figsize=(figsize[0]*len(axis_ref_bias_list), figsize[1]*nrows))
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    subfigs = fig.subfigures(nrows=nrows)

    for axis, subfig in zip(axis_list, subfigs[0:nrows//2]):
        axis_remaining = list(set(axis_list) - {axis})

        axes = subfig.subplots(ncols=len(axis_ref_bias_list), nrows=1)
        for ax, (axis_ref, axis_bias) in zip(axes, axis_ref_bias_list):
            score = U.r2_score(y_true, y_pred, axis=axis, axis_ref=axis_ref, axis_bias=axis_bias)
            plot_dimensional_score(score, ax=ax)
        [ax.set_xlabel(axislabel_list[axis_remaining[1]]) for ax in axes]
        [ax.set_ylabel(axislabel_list[axis_remaining[0]]) for ax in axes]

        axis_name = axislabel_list[axis]
        subfig.suptitle(f"Axis: ({axis_name})", y=suptitle_y, fontsize=subfigtitlesize)

    # 2D (1D result)
    for (i, j), subfig in zip(it.combinations(axis_list, 2), subfigs[nrows//2:]):
        axis_remaining = list(set(axis_list) - {i, j})

        axes = subfig.subplots(ncols=len(axis_ref_bias_list), nrows=1)
        for ax, (axis_ref, axis_bias) in zip(axes, axis_ref_bias_list):
            score = U.r2_score(y_true, y_pred, axis=(i,j), axis_ref=axis_ref, axis_bias=axis_bias)
            plot_dimensional_score(score, ax=ax)

        [ax.set_xlabel(axislabel_list[axis_remaining[0]]) for ax in axes]
        [ax.set_yticks([]) for ax in axes]

        axis_names = axislabel_list[i],axislabel_list[j]
        subfig.suptitle(f"Axis: ({axis_names[0]},{axis_names[1]})", y=suptitle_y, fontsize=subfigtitlesize)
    
    axes = np.array(fig.axes).reshape(nrows, len(axis_ref_bias_list))
    [ax.set_title(f"Ref: {axis_ref}, Bias: {axis_bias}", y=1.18) for ax, (axis_ref, axis_bias) in zip(axes[0], axis_ref_bias_list)]
    [add_colorbar(ax) for ax in axes[:,-1]]

    return fig, axes

def plot_r2_axis(data_dict, axis, axis_ref_bias_list):
    fig = plt.figure(figsize=(figsize[0]*len(axis_ref_bias_list), figsize[1]*len(data_dict)))
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    subfigs = fig.subfigures(nrows=len(data_dict))

    for subfig, (data_name, data) in zip(subfigs, data_dict.items()):
        axes = subfig.subplots(ncols=len(axis_ref_bias_list), nrows=1)
        y_true, y_pred = data['y_true'], data['y_pred']
        
        for ax, (axis_ref, axis_bias) in zip(axes, axis_ref_bias_list):
            score = U.r2_score(y_true, y_pred, axis=axis, axis_ref=axis_ref, axis_bias=axis_bias)
            plot_dimensional_score(score, ax=ax)

        axis_remaining = list(set(axis_list) - {axis})
        [ax.set_xlabel(axislabel_list[axis_remaining[1]]) for ax in axes]
        [ax.set_ylabel(axislabel_list[axis_remaining[0]]) for ax in axes]

        axis_name = axislabel_list[axis]
        subfig.suptitle(f"Data name: {data_name}", y=suptitle_y, fontsize=subfigtitlesize)

    axes = np.array(fig.axes).reshape(len(data_dict), len(axis_ref_bias_list))
    [ax.set_title(f"Ref: {axis_ref}, Bias: {axis_bias}", y=1.18) for ax, (axis_ref, axis_bias) in zip(axes[0], axis_ref_bias_list)]
    [add_colorbar(ax) for ax in axes[:,-1]]

    return fig, axes

def plot_dimensional_score(score, ax=None):
    if isinstance(score, float):
        score = np.array([[score]])
    elif score.ndim == 1:
        score = score.reshape(1, -1)
        ax.set_yticks([])  # Hide y-ticks if score is 1D
    im = ax.matshow(score, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    # fig.colorbar(im, ax=ax, label='R^2 Score')
    # ax.set_title('Dim-$R^2$')
    return im

def get_axislabel(axis):
    if isinstance(axis, Iterable):
        return tuple(axislabel_dict[axis_] for axis_ in axis)
    else:
        return axislabel_dict[axis]

# %%
'''
Dim-R2 across data (Time, Neuron), across all data types
'''
axis_ref_bias_list = [(0,0), (1, 1), ((0,1), (0,1)), ((0,1), 1), ((0,1),0)]
fig, axes = plot_r2_axis(data_dict, axis=0, axis_ref_bias_list=axis_ref_bias_list)
fig.savefig(path_save / f'r2_axis_0.png', dpi=300, bbox_inches='tight')

axis_ref_bias_list = [(None,None), (1, 1), ((0,1), (0,1)), ((0,1), 1), ((0,1),0)]
fig, axes = plot_r2_axis(data_dict, axis=2, axis_ref_bias_list=axis_ref_bias_list)
fig.savefig(path_save / f'r2_axis_2.png', dpi=300, bbox_inches='tight')

# %%
'''
Comprehensive Dim-R2 (All dimensional combinations), per data types
'''
for data_name, data in data_dict.items():
    path_save_data = path_save / data_name

    y_true, y_pred = data['y_true'], data['y_pred']

    fig, axes = plot_r2_combinations(y_true, y_pred, axis_ref_bias_list=axis_ref_bias_list)
    fig.suptitle(f"{data_name}", y=1.02)
    fig.savefig(path_save_data / f'r2_axis_ref_bias.png', dpi=300, bbox_inches='tight')

# %%
'''
Extensive Dim-R2 (All dimensional combinations + RSS/TSS), per data types
'''
for data_name, data in data_dict.items():
    path_save_data = path_save / data_name

    y_true, y_pred = data['y_true'], data['y_pred']

    for axis_ref, axis_bias in axis_ref_bias_list:
        try:
            fig, axes = plot_r2_combinations_extensive(y_true, y_pred, axis_ref=axis_ref, axis_bias=axis_bias)
            fig.suptitle(f"{data_name} | axis_ref: ({axis_ref}) | axis_bias: ({axis_bias})", y=1.02)
            fig.savefig(path_save_data / f"r2_extensive_ref_{axis_ref}_bias_{axis_bias}.png", dpi=300, bbox_inches='tight')
        except AssertionError as e:
            print(e)

# %%
# plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.tab10.colors)

# %%
cmap = plt.get_cmap('tab10')
path_save = T.Path()

def plot_individual(t, y_true, y_pred, old_r2, dim_r2, r2_per_channel, ylim=None, figsize=None):
    channel_i = range(len(y_true[1]))
    var_per_channel_y_true = np.var(y_true, axis=0)
    var_per_channel_y_pred = np.var(y_pred, axis=0)
    figsize = (6, y_true.shape[1]*1.2) if figsize is None else figsize

    # fig, axes = plt.subplots(ncols=y_true.shape[1], nrows=2, figsize=(C*2, 5), sharex=True, sharey=True)
    fig, axes = plt.subplots(ncols=2, nrows=y_true.shape[1], figsize=figsize, sharex=True, sharey=True)
    for y_true_, y_pred_, r2, var_true, var_pred, c, i, axes_ in zip(y_true.T, y_pred.T, r2_per_channel, var_per_channel_y_true, var_per_channel_y_pred, cmap.colors, channel_i, axes):
        # axes_[0].plot(y_true_, t, c=c, label=f'C{i}-$R^2$:{r2:.2f},Var:{var_true:.2f}')
        # axes_[1].plot(y_pred_, t, c=c)
        var_true, var_pred = np.round(var_true, 2), np.round(var_pred, 2)

        axes_[0].plot(t, y_true_, c=c, label=f'Var:{var_true:.2f}')
        axes_[1].plot(t, y_pred_, c=c, label=f'Var:{var_pred:.2f}')
        axes_[0].legend(loc='upper right')
        axes_[1].legend(loc='upper right')
        axes_[0].set_ylabel(f'C{i}-$R^2$:{r2:.2f}')

        if ylim is None:
            ylim = np.abs(axes_[0].get_ylim()).max()
            axes_[0].set_ylim(-ylim,ylim)
        else:
            axes_[0].set_ylim(-ylim,ylim)

    # axes[-1,0].set_xlabel('y_true')
    # axes[-1,1].set_xlabel('y_pred')  
    fig.suptitle(rf"Mean $R^2$: {old_r2:.2f} / Dim-$R^2$ {dim_r2:.2f}", y=0.93)
    return fig, axes

def plot(t, y_true, y_pred, old_r2, dim_r2, r2_per_channel, ylim=None):
    # color = ['grey' if r2>0.99 else plt.cm.tab10.colors[i] for i, r2 in enumerate(r2_per_channel)]
    alpha = [0.5 if r2>0.99 else 1 for r2 in r2_per_channel]
    line = ['-' if r2>0.99 else '--' for r2 in r2_per_channel]
    channel_i = range(len(y_true[1]))

    var_per_channel = np.var(y_true, axis=0)
    fig, axes = plt.subplots(nrows=2, figsize=(10, 10), sharex=True)
    for y_true_, y_pred_, r2, var, alpha, c, l, i in zip(y_true.T, y_pred.T, r2_per_channel, var_per_channel, alpha, cmap.colors, line, channel_i):
        axes[0].plot(t, y_true_, alpha=alpha, c=c, linestyle=l, label=f'C{i}-$R^2$:{r2:.2f},Var:{var:.2f}')
        axes[1].plot(t, y_pred_, alpha=alpha, c=c, linestyle=l)

    # axes[0].plot(t, y_true, alpha=alpha, label=[f'Channel {i}-R2:{r2:.2f},Var:{var:.2f}' for i, (r2, var) in enumerate(zip(r2_per_channel, var_per_channel))])
    axes[0].legend(loc='upper right')
    axes[0].set_ylabel('y_true')
    if ylim is None:
        ylim = np.abs(axes[0].get_ylim()).max()
        axes[0].set_ylim(-ylim,ylim)
    else:
        axes[0].set_ylim(-ylim,ylim)
    # axes[1].plot(t, y_pred, alpha=alpha)
    axes[1].set_ylabel('y_pred')
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_xlabel('time')
    fig.suptitle(rf"Mean $R^2$: {old_r2:.2f} / Dim-$R^2$ {dim_r2:.2f}", y=0.93)
    return fig, axes


            
# plt.plot(noise(y_true[:,0], noisetype='linear_increasing'))
# plt.plot(noise(y_true[:,0], noisetype='linear_decreasing'))
# plt.plot(noise(y_true[:,0], noisetype='constant'))

# %%
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

    old_r2 = r2_score_sklearn(y_true, y_pred, multioutput='uniform_average')
    dim_r2 = U.r2_score(y_true, y_pred)
    r2_per_channel1 = U.r2_score(y_true, y_pred, axis=0)
    r2_per_channel2 = r2_score_sklearn(y_true, y_pred, multioutput='raw_values')
    assert np.allclose(r2_per_channel1, r2_per_channel2), f"r2_score: {r2_per_channel1}, r2_score_sklearn: {r2_per_channel2}"

    fig, axes = plot_individual(t=t, y_true=y_true, y_pred=y_pred, old_r2=old_r2, dim_r2=dim_r2, r2_per_channel=r2_per_channel1, ylim=ylim)
    # fig, axes = plot(t=t, y_true=y_true, y_pred=y_pred, old_r2=old_r2, dim_r2=dim_r2, r2_per_channel=r2_per_channel1, ylim=ylim)

    for i, ax in enumerate(axes.T[0]):
        ylabel = 'Noise' if i in l_noise else 'Neuron'
        ylabel += '\n' + rf'$R^2$:{r2_per_channel1[i]:.2f}'
        ax.set_ylabel(ylabel, fontsize=12)
    
    for i, axes_ in enumerate(axes):
        if i in l_noise:
            [ax.get_lines()[0].set_color(d_colors['noise']) for ax in axes_]
        else:
            [ax.get_lines()[0].set_color(d_colors['regular']) for ax in axes_]
        [ax.legend(loc='upper right') for ax in axes_]

    # axes[0,0].set_xlabel(f'$y_{{true}}$ (Noise Neuron var={var1})\nTime')
    # axes[0,1].set_xlabel(f'$y_{{pred}}$ (Noise Neuron var={var2})\nTime')
    axes[0,0].set_xlabel(f'$y_{{true}}$\nTime')
    axes[0,1].set_xlabel(f'$y_{{pred}}$\nTime')

    axes[0,0].xaxis.set_label_position('top')
    axes[0,1].xaxis.set_label_position('top')
    axes[-1,0].set_xticks([])  # Hide x-ticks for the last row
    axes[-1,1].set_xticks([])  # Hide x-ticks for the last row
    # axes[-1,0].xaxis.set_visible(False)
    # axes[-1,1].xaxis.set_visible(False)

    fig.suptitle(rf"Mean $R^2$: {old_r2:.2f} / Dim-$R^2$ {dim_r2:.2f}", y=0.95)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(path_save/f"r2_score_{var1}_{var2}.png", dpi=300, bbox_inches='tight')
    # plt.close(fig)

# %%
D = 1000
N = 100
C = 5
    # bias = np.random.randn(1)*std_bias
trial_bias=False
# trial_bias=True

# t, y_true, y_pred = make_signals(N, C)
t, y_true, y_pred = make_signals(N, C, stagger_phase=False)
y_true, y_pred = y_true*2, y_pred*2
y_true_og, y_pred_og = y_true.copy(), y_pred.copy()
ylim = np.abs(y_true).max() * 1.09

var1, var2 = 0.01, 0.1
std_bias = 0.5

var_true = y_true[:,0].var()
std1, std2 = 0.5*var_true, 0.5*var_true
# var_targets = [0.01, 0.1, 1.0]
d_noise_add = {1: 'linear_increasing', 2: 'linear_decreasing', 3: 'constant'}
l_noise = [4]
C_i_noise = list(d_noise_add.keys()) + l_noise

l_y_true, l_y_pred = [], []
for i in range(D):

    y_true, y_pred = y_true_og.copy(), y_pred_og.copy()
    if trial_bias:
        bias = np.random.randn(1)*std_bias
    else:
        bias = 0
    y_true, y_pred = y_true + bias, y_pred + bias

    for C_i in d_noise_add.keys():
        noise1, noise2 = noise(y_true[:,C_i], noisetype=d_noise_add[C_i]), noise(y_pred[:,C_i], noisetype=d_noise_add[C_i])
        # y_true[:,C_i] += noise1 * np.sqrt(var1)
        # y_pred[:,C_i] += noise2 * np.sqrt(var2)
        y_true[:,C_i] += noise1 * std1
        y_pred[:,C_i] += noise2 * std2

    for C_i in l_noise:
        y_true[:,C_i], y_pred[:,C_i] = np.random.randn(*y_true[:,C_i].shape), np.random.randn(*y_pred[:,C_i].shape)
    # y_true[:,i], y_pred[:,i] = np.random.randn(*y_true[:,i].shape)*noise1, np.random.randn(*y_pred[:,i].shape)*noise2
    # y_true_var, y_pred_var = np.var(y_true[:,C_i_noise], axis=0), np.var(y_pred[:,C_i_noise], axis=0)
    y_true_var, y_pred_var = np.var(y_true[:,l_noise], axis=0), np.var(y_pred[:,l_noise], axis=0)
    y_true[:,l_noise], y_pred[:,l_noise] = y_true[:,l_noise]*np.sqrt(var1/y_true_var), y_pred[:,l_noise]*np.sqrt(var2/y_pred_var)
    l_y_true.append(y_true), l_y_pred.append(y_pred)

y_true, y_pred = np.stack(l_y_true), np.stack(l_y_pred)
print(y_true.shape, y_pred.shape)

# %%
dim_r2 = T.sklearn.metrics.r2_score(y_true, y_pred, axis=0)
plt.matshow(dim_r2.T, cmap='Blues', vmin=0, vmax=1, aspect='auto')

# %%

old_r2 = r2_score_sklearn(y_true[0], y_pred[0], multioutput='uniform_average')
dim_r2_ = r2_score(y_true[0], y_pred[0])
r2_per_channel1 = r2_score(y_true[0], y_pred[0], axis=0)

# %%
figsize = (6, C*1.5)
fig, axes = plot_individual(t=t, y_true=y_true[0], y_pred=y_pred[0], old_r2=old_r2, dim_r2=dim_r2_, r2_per_channel=r2_per_channel1, ylim=ylim, figsize=figsize)
ylim = np.abs(y_true[0]).max() * 1.09
[ax.set_ylim(-ylim, ylim) for ax in axes.flatten()]
[ax.legend().remove() for ax in axes.flatten()]
for i, ax in enumerate(axes.T[0]):
    ylabel = rf'$R^2$:{r2_per_channel1[i]:.2f}'
    ax.set_ylabel(ylabel)

[ax.get_lines()[0].set_color(d_colors['regular']) for ax in axes.flatten()]
axes[0,0].set_xlabel(f'$y_{{true}}$')
axes[0,1].set_xlabel(f'$y_{{pred}}$')
axes[0,0].xaxis.set_label_position('top')
axes[0,1].xaxis.set_label_position('top')
# axes[-1,0].xaxis.set_visible(False)
# axes[-1,1].xaxis.set_visible(False)
[ax.set_xticks([]) for ax in axes.flatten()]
# [ax.xaxis.set_visible(False) for ax in axes.flatten()]
fig.subplots_adjust(hspace=0.1, wspace=0.05)

fig.suptitle(rf"Mean $R^2$: {old_r2:.2f} / Dim-$R^2$ {dim_r2_:.2f}", y=0.95)
fig.patch.set_alpha(0.0)
fig.savefig(path_save/f"r2_score_timevarying.png", dpi=300, bbox_inches='tight')
    # plt.close(fig)
# %%
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.matshow(dim_r2.T, cmap='Blues', vmin=0, vmax=1, aspect='auto')
fig.suptitle(f'(Trials, Time, Neuron) = {y_true.shape}', y=1.07)
cbar = fig.colorbar(im)
cbar.set_label('Dim-$R^2$', rotation=90, labelpad=25)
ax.set_yticklabels(np.arange(y_true.shape[2]+1))
ax.set_ylabel('Neuron')
ax.set_xlabel('Time')
ax.xaxis.set_label_position('top')
fig.patch.set_alpha(0.0)
fig.savefig(path_save/f"dim_r2.png", dpi=300, bbox_inches='tight')

# %%
fig, ax = plt.subplots(figsize=(3,5))
im = ax.matshow(dim_r2.T, cmap='Blues', vmin=0, vmax=1, aspect='auto')
cbar = fig.colorbar(im)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
fig.patch.set_alpha(0.0)
fig.savefig(path_save/f"dim_r2.png", dpi=300, bbox_inches='tight')

# fig, axes = plot(t=t, y_true=y_true, y_pred=y_pred, old_r2=old_r2, dim_r2=dim_r2, r2_per_channel=r2_per_channel1, ylim=ylim)
# axes[0].set_ylabel(f'$y_{{true}}$ (Noise var={var1})')
# axes[1].set_ylabel(f'$y_{{pred}}$ (Noise var={var2})')
# fig.suptitle(rf"Mean $R^2$: {old_r2:.2f} / Dim-$R^2$ {dim_r2:.2f}", y=0.95)
# fig.savefig(path_save/f"r2_score_{var1}_{var2}.png", dpi=300, bbox_inches='tight')
# plt.close(fig)

# %%
N = 100
C = 100
t, y_true, y_pred = make_signals(N, C)
y_true_og, y_pred_og = y_true.copy(), y_pred.copy()
print(y_true.shape)

var_targets = [0.01, 0.1, 1.0]
p_noise_channels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
n_repeat = 100
l_results = []
for (var1, var2), p_noise in tqdm(it.product(it.product(var_targets, repeat=2), p_noise_channels)):
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
df_agg = df_results.groupby(['var1', 'var2', 'p_noise']).agg({'old_r2': ['mean', 'std'], 'old_r2_var':['mean','std'], 'dim_r2': ['mean', 'std']}).reset_index()
# df_agg = df_results.groupby(['var1', 'var2', 'Ratio of Noise Channels']).agg({'Mean R2': ['mean', 'std'], 'Dim-R2': ['mean', 'std']}).reset_index()
df_agg
writer = pd.ExcelWriter(path_save/'r2_score_results.xlsx', engine='openpyxl')
df_results.to_excel(writer, sheet_name='all')
df_agg.to_excel(writer, sheet_name='agg')
writer.close()

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
    'old_r2': 'Mean $R^2$',
    'old_r2_var': 'Mean $R^2$ (Variance Weighted)',
    'dim_r2': 'Dim-$R^2$',
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

    
    if gaussian1 and gaussian2:

        stat, p = stats.levene(df_group['old_r2'], df_group['dim_r2'])
        equal_var = p > 0.05
        t1 = TableOne(df_score, normal_test=True, groupby='R2 Type', pval=True, continuous=['R2 Score'], htest_name=True, ttest_equal_var=equal_var)

        if p > 0.05:
            stat, p = stats.ttest_ind(df_group['old_r2'], df_group['dim_r2'])
        else:
            stat, p = stats.mannwhitneyu(df_group['old_r2'], df_group['dim_r2'])
    p2
    df_group
df_results_var1 = df_results[df_results['var1'] == var1]
df_scores=df_results_var1[['old_r2','dim_r2']].unstack()
df_scores = df_scores.reset_index()
df_results_melt = df_results.melt(id_vars=['var1', 'var2', 'p_noise'], value_vars=['old_r2', 'dim_r2'], var_name='R2 Type', value_name='R2 Score')
df_results_melt_var1 = df_results_melt[df_results_melt['var1'] == var1]

t1 = tableone(df_results_melt_var1, normal_test=True, groupby='R2 Type', pval=True, continuous=['R2 Score'], nonnormal=['R2 Score'])
t1 = TableOne(df_results_melt_var1, normal_test=True, columns = ['R2 Score'], groupby='R2 Type', pval=True, continuous=['R2 Score'], htest_name=True)
t1.cont_describe
tableone
TableOne
t1.htest_table
plt.plot()
help(tableone)
help(TableOne)

# %%
'''
Plot sample waveforms
'''
N=100
C=4
t, y_true, y_pred = U.make_signals(N, C)
ylim = np.abs(y_true).max() * 1.09

y_true[:,-1:] = np.random.randn(*y_true[:,-1:].shape)*0.1

colors = ['#D3896B','#203864','#91BCE3','#595959']
path_save = T.Path()

# %%
fig, axes = plt.subplots(ncols=C, nrows=1, figsize=(C, 4), sharex=True, sharey=True)
for y_true_, ax, c in zip(y_true.T, axes, colors):
    ax.plot(y_true_, t, c=c)
    ax.set_xlim(-ylim, ylim)
    
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # ax.set_xticks(np.linspace(0, 2*np.pi, 5))
    # ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(0, 2*np.pi, 5)])
    # ax.set_xlabel('time')
    # ax.set_ylabel('y_true')
    # ax.legend(loc='upper right')
fig.savefig(path_save/f"sample_waveform_{C}.png", dpi=300, bbox_inches='tight')

# %%
# Save individual channels as well
for i, (y_true_, c) in enumerate(zip(y_true.T, colors)):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1, 4), sharex=True, sharey=True)
    ax.plot(y_true_, t, c=c)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlim(-ylim, ylim)

    fig.savefig(path_save/f"sample_waveform_{C}_{i}.png", dpi=300, bbox_inches='tight')    

# %%
x=[0.3,0.4,0.9,0.8,0.01,0.05,-80,-100]\
# %%
r2_score(y_true, y_pred, axis=0, multioutput='raw_values') # R2 across channel dimension, collapse time and trial
r2_score(y_true, y_pred, axis=(0,2), multioutput='raw_values') # R2 across trial and channel dimension, collapse time

r2_score(y_true, y_pred, axis=0, multioutput='raw_values').shape # R2 across trial dimension
r2_score(y_true, y_pred, axis=1, multioutput='raw_values').shape # R2 across time dimension
r2_score(y_true, y_pred, axis=2, multioutput='raw_values').shape # R2 across channel dimension
r2_score(y_true, y_pred, axis=(0,1), multioutput='raw_values').shape # R2 across trial and time dimension
r2_score(y_true, y_pred, axis=(0,2), multioutput='raw_values').shape # R2 across trial and channel dimension 
r2_score(y_true, y_pred, axis=(1,2), multioutput='raw_values').shape # R2 across time and channel dimension 

# %%
y_pred.shape # (Trials, time, channel)
r2_score(y_true, y_pred, axis=1, multioutput='raw_values') # R2 across trial dimension
r2_score(y_true, y_pred, axis=1, multioutput='raw_values').mean() # R2 across trial dimension

r2_score(y_true, y_pred, axis=(0,1), multioutput='raw_values') # R2 across trial & time dimension

# %%
'''
Plot experiment result
'''
path_file = 'C:\\Users\\jsyoo\\Mirror\\UNC Chapel Hill\\Hantman Lab\\exp\\rnn_modeling\\analysis\\rnn\\full_horizon2\\sweep_result.p.xlsx'
df_all = pd.read_excel(path_file)
d_rename = {'r2': 'Dim-$R^2$', 'r2_N_sklearn_uniform': 'Mean $R^2$'}
df_all.rename(columns=d_rename, inplace=True)

# %%
df_exp_success = df_all.query('`data.session_i`==3 & task=="spiketrain_hand" & `preprocess.filter.sigma` == 25 & `preprocess.standardize`==True & `preprocess.pca`==False & `network.nonlinearity`=="tanh"')
df_exp_fail = df_all.query('`data.session_i`==3 & task=="spiketrain_hand" & `preprocess.filter.sigma` == 25 & `preprocess.standardize`==False & `preprocess.pca`==False & `network.nonlinearity`=="relu"')
df_exp_fail
# %%
columns = list(d_rename.values())
figsize = (12,4)
ncols = 3
margin = 0.1
def plot_boxplot(df, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(2,4))

    mean = df[columns].values.mean(0)
    ax.boxplot(df[columns].values, widths=0.3*figsize[0]/(ncols*2))
    ax.set_title(f'N=({len(df)})')
    ax.set_ylabel('$R^2$ Value')
    ax.set_ylim(0-margin, 1+margin)

    xticklabels = [f'{c}\n($\\mu$: {m:.1E})' if np.abs(m) > 1e2 else f'{c}\n($\\mu$: {m:.2f})' for c, m in zip(columns, mean)]

    ax.set_xticklabels(xticklabels)
    ax.grid(axis='y')
    return ax
    ax.get_xticklabels()

# %%
df_exp_fail[columns].values.min(0)
fig, axes = plt.subplots(ncols=ncols, figsize=figsize)
yticks = np.linspace(0,1,5)
y=1.05

plot_boxplot(df_exp_success, ax=axes[0])
axes[0].set_title(f'Accurate RNNs\n(N={len(df_exp_success)})', y=y)
axes[0].set_yticks(yticks)

plot_boxplot(df_exp_fail, ax=axes[1])
axes[1].set_title(f'Undertrained RNNs\n(N={len(df_exp_success)})', y=y)
axes[1].set_yticks(yticks)

plot_boxplot(df_exp_fail, ax=axes[2])
axes[2].set_title(f'Undertrained RNNs\n(N={len(df_exp_success)})', y=y)

axes[2].set_ylim(df_exp_fail[columns].values.min()*1.05, 1e36)
axes[2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
axes[2].yaxis.get_offset_text().set_x(-0.24)  # Move offset text to the left
fig

# axes[2].set_yscale('log')
# yaxis.scale('log')
# [d for d in dir(axes[2].yaxis) if 'scale' in d]
# [d for d in dir(axes[2]) if 'scale' in d]
# axes[2].yaxis.get_scale()

fig.tight_layout()
fig.patch.set_alpha(0.0)
# %%
fig.savefig(path_save/f"r2_comparison.png", dpi=300, bbox_inches='tight')

# %%
