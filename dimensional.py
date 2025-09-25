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
T = 100
C = 5

t, y_true, y_pred = U.make_signals(T, C, stagger_phase=False)
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
    'No Bias': {
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

# Add mean prediction of y_true to each dataset as a control
for data_name, data in data_dict.items():
    y_true, y_pred = data['y_true'], data['y_pred']    
    y_pred_mean = np.broadcast_to(y_true.mean(axis=1, keepdims=True), y_true.shape)
    data_dict[data_name]['y_pred_mean'] = y_pred_mean

# %%
for data_name, data in data_dict.items():
    path_save_data = path_save / data_name
    path_save_data.mkdir(parents=True, exist_ok=True)

# %%
'''
Sample waveforms
'''
def plot_sample(y, xlim_sync=False, figsize_y=2.5):
    '''
    y: (time, neuron)
    '''
    fig, axes = plt.subplots(ncols=y.shape[1], figsize=(2*y.shape[1], figsize_y), sharey=True)
    yticks = np.arange(y.shape[0])
    for i, (ax, y_) in enumerate(zip(axes, y.T)):
        ax.plot(y_, yticks)
        ax.invert_yaxis()
        ax.set_xlabel(f'N{i} ($\mu$={int(np.round(y_.mean())):d})')

    axes[0].set_ylabel('Time')

    if xlim_sync:
        xlim = np.abs(np.array([ax.get_xlim() for ax in axes])).max()
        [ax.set_xlim(-xlim, xlim) for ax in axes]
    else:
        xlim = axes[0].get_xlim()
        xlim_mean = np.round(y.mean(axis=0)).astype(int)
        xlim_range = (xlim[1] - xlim[0])*xlim_amplifier
        xlim = np.stack([xlim_mean - xlim_range/2, xlim_mean + xlim_range/2], axis=1)
        [ax.set_xlim(xlim_ ) for ax, xlim_ in zip(axes, xlim)]
        [ax.set_xticks([xlim_mean_ -2, xlim_mean_, xlim_mean_ + 2]) for ax, xlim_mean_ in zip(axes, xlim_mean)]

    return fig, axes

data_i = 1
xlim_amplifier = 1.05
n_samples = 3

for data_name in data_dict.keys():
    y = data_dict[data_name]['y_true'][data_i]
    fig, axes = plot_sample(y, xlim_sync=False, figsize_y=2)
    fig.suptitle(data_name, y=1.02)
    fig.savefig(path_save/f"sample_{data_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')


xlim_max = np.abs(np.array([ax.get_xlim() for ax in axes])).max()

for data_name in data_dict.keys():
    
    for i in range(n_samples):
        y = data_dict[data_name]['y_true'][data_i+i]
        fig, axes = plot_sample(y, xlim_sync=True)
        if data_name == 'No Bias':
            [ax.set_xlim(-xlim_max, xlim_max) for ax in axes]

        if i!=0: # Only keep xlabel for the first trial
            [ax.set_ylabel('') for ax in axes]
        # fig.suptitle(data_name)
        fig.savefig(path_save/f"sample_{data_name.lower().replace(' ', '_')}_sync_{i}.png", dpi=300, bbox_inches='tight')

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

def add_colorbar(ax, mappable=None, width=0.02, pad=0.02, divide=False):
    """
    Add a colorbar to the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The colorbar added to the axes.
    """
    fig = ax.figure
    if mappable is None:
        if ax.images:
            mappable = ax.images[0]
        else:
            raise ValueError("No mappable found in the axes. Please provide a mappable or ensure the axes contain an image.")

    if divide:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=width, pad=pad)
    else:        
        bbox = ax.get_position()
        cax = fig.add_axes([bbox.x1 + pad, bbox.y0, width, bbox.height])
    cbar = fig.colorbar(mappable, cax=cax)
    # cbar = ax.figure.colorbar(cm.ScalarMappable(norm=ax.images[0].norm, cmap=ax.images[0].cmap), cax=cax)
    # cbar = ax.figure.colorbar(ax.images[0], ax=ax, location='right', pad=0.15)

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
[Main text figure]
2D Dim-R2 with/without argument adjustments + Baseline Dim-R2 of Mean prediction
across all 3 data types
'''
figsize_multiplier = 0.7
# titles = ['$y_{pred}$\nAxis_ref=Time', '$y_{pred}$\nAxis_ref=Trial (Default)', '$\\bar{y}_{true}$\nAxis_ref=Time', '$\\bar{y}_{true}$\nAxis_ref=Trial (Default)']
titles = ['$y_{pred}$\nAxis_ref=Axis (Trial)', '$\\bar{y}_{true}$\nAxis_ref=Axis (Trial)', '$y_{pred}$\nAxis_ref=Time', '$\\bar{y}_{true}$\nAxis_ref=Time']
for data_name, data in data_dict.items():
    path_save_data = path_save / data_name

    y_true, y_pred, y_pred_mean = data['y_true'], data['y_pred'], data['y_pred_mean']

    r2_basic = U.r2_score(y_true, y_pred, axis=0)
    r2_adjusted = U.r2_score(y_true, y_pred, axis=0, axis_ref=1, axis_bias=1)
    r2_baseline = U.r2_score(y_true, y_pred_mean, axis=0)
    r2_baseline_adjusted = U.r2_score(y_true, y_pred_mean, axis=0, axis_ref=1, axis_bias=1)

    # scores = [r2_adjusted, r2_basic, r2_baseline_adjusted, r2_baseline]
    scores = [r2_basic, r2_baseline, r2_adjusted, r2_baseline_adjusted]
    fig, axes = plt.subplots(ncols=4, figsize=(figsize[0]*len(scores)*figsize_multiplier, figsize[1]*figsize_multiplier*1.5), sharey=True)
    for ax, score, title in zip(axes, scores, titles):
        im = plot_dimensional_score(score, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Neuron')
        ax.xaxis.set_ticks_position('bottom')
    axes[0].set_ylabel('Time')
    
    fig.suptitle(f"{data_name}", y=0.94)
    fig.tight_layout()
    add_colorbar(axes[-1])
    fig.savefig(path_save_data / f'dim_r2_{data_name}.png', dpi=300, bbox_inches='tight')

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
