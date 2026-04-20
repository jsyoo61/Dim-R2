# %%
from collections.abc import Iterable
import warnings

import numpy as np

import tools.numpy
import matplotlib.pyplot as plt

# %%
plt.rcParams['font.size'] = 13
plt.rcParams['mathtext.default']='regular'
d_colors = {
    'regular': '#88B6E0',
    'noise':'#595959'
    }

# %%
'''
R2 score functions
'''
def r2_score_old(y_true, y_pred, axis=None, axis_ref=None, multioutput='raw_values', force_finite=True):
    """
    R^2 score for multidimensional predictions.
    collapses all axes except the specified axis.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    axis: int or iterable of int, default=None
        Axis to collapse.
        If None, collapses all axes.
        
    multioutput : Reference to `sklearn.metrics.r2_score`
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

        Note:
        - default is 'raw_values', which is different from sklearn.
        - when multioutput is uniform_average or variance_weighted, return value is a single float even if axis is specified.
          The axis specifies the feature dimensions to average over.
    
    Returns
    -------
    z : np.ndarray
        if axis is specified, returns an array of shape (y_true.shape[axis],)
    """
    assert y_true.shape == y_pred.shape, f"y_true and y_pred must have the same shape, received: {y_true.shape} and {y_pred.shape}"
    if axis is None:
        axis = list(range(y_true.ndim)) # Collapse all dimensions
        axis_was_none = True
    elif not isinstance(axis, Iterable):
        axis = [axis]
        axis_was_none = False
    else:
        axis = list(axis)
        axis_was_none = False

    if len(axis) > y_true.ndim:
        raise ValueError("Axis is greater than the number of dimensions of y_true and y_pred")

    dim_collapse = axis
    dim_final = list(set(range(y_true.ndim)).difference(axis))
    shape_final = np.array(y_true.shape)[dim_final] if len(dim_final)!=0 else (1,)

    y_true_collapsed = np.transpose(y_true, (*dim_collapse, *dim_final)).reshape(-1, np.prod(shape_final)) # Move axis to the end, and flatten the rest
    y_pred_collapsed = np.transpose(y_pred, (*dim_collapse, *dim_final)).reshape(-1, np.prod(shape_final))

    score = r2_score_sklearn(y_true_collapsed, y_pred_collapsed, multioutput='raw_values')

    if type(score) == float and np.isnan(score).item():
        warnings.warn("R2 score is a single NaN, shape matching with NaN")
        score = np.full(shape_final, np.nan)
        return score

    if multioutput == 'raw_values':
        score = score.reshape(*shape_final) if not axis_was_none else score[0]
        return score
    elif multioutput == 'uniform_average':
        return score.mean()
    elif multioutput == 'variance_weighted':        
        return np.average(score, weights=np.var(y_true_collapsed, axis=0))
    else:
        raise ValueError("multioutput must be one of ['raw_values', 'uniform_average', 'variance_weighted']")

def r2_score_plot(y_true, y_pred, axis=None, axis_ref=None, axis_bias=None, multioutput='raw_values', force_finite=True, fig=None, axes=None, axislabels=None):
    """
    R^2 score for multidimensional predictions.
    collapses all axes except the specified axis.

    Computes 1 - RSS / TSS, where RSS is the residual sum of squares and TSS is the total sum of squares.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    axis: int or iterable of int, default=None
        Axis to collapse.
        If None, collapses all axes.
        
    multioutput : Reference to `sklearn.metrics.r2_score`
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

        Note:
        - default is 'raw_values', which is different from sklearn.
        - when multioutput is uniform_average or variance_weighted, return value is a single float even if axis is specified.
          The axis specifies the feature dimensions to average over.
    
    Returns
    -------
    z : np.ndarray
        if axis is specified, returns an array of shape (y_true.shape[axis],)
    """

    # Residual Sum of Squares (RSS) and Total Sum of Squares (TSS)
    RS = (y_true - y_pred)**2 # Residual Square (RS)
    RSS = np.sum(RS, axis=axis, keepdims=True)

    if axis_ref is None:
        axis_ref = axis
    if axis_bias is None:
        axis_bias = axis_ref

    axis_ref_set = {axis_ref} if not isinstance(axis_ref, Iterable) else set(axis_ref)
    axis_set = {axis} if not isinstance(axis, Iterable) else set(axis)
    axis_bias_set = {axis_bias} if axis_bias is not None and not isinstance(axis_bias, Iterable) else set(axis_bias) if axis_bias is not None else set()

    assert axis_bias_set.issubset(axis_ref_set), f'axis_bias ({axis_bias}) must be a subset of axis_ref ({axis_ref})' # If axis_bias is not a subset of axis_ref, expand axis_ref to include axis_bias

    # Dimensions of TSS must be smaller than RSS, so mean/sum over axis_ref and axis
    axis_sum = axis
    axis_mean = tuple(axis_ref_set - axis_set) # Average over axis_ref - axis

    if not axis_set.issubset(axis_ref_set): # If axis is not a subset of axis_ref, expand axis_ref to include axis
        warnings.warn(f"axis {axis} is not a subset of axis_ref {axis_ref}, TSS sums over axis {axis_sum} and averages over remaining axis_ref {axis_mean}")

    # axis_bias used to compute the mean of y_true
    y_mean = np.mean(y_true, axis=axis_bias, keepdims=True)
    TS = (y_true - y_mean)**2 # Total Square (TS)

    # axis_ref used to aggregate additional dimensions (average) additional to axis
    TSS = np.mean(TS, axis=axis_mean, keepdims=True)
    TSS = np.sum(TSS, axis=axis_sum, keepdims=True)
    
    score = 1 - RSS / TSS
    score = np.squeeze(score, axis=axis) # Collapse the axis_ref dimension

    if force_finite:
        score[np.isnan(score)] = 1
        score[np.isinf(score)] = 0 # -Inf means no fit, so set to 0

    if axis is None:
        score = score.item()

    # Temporary plot functions
    assert y_true.ndim<=3

    if fig is None or axes is None:
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))
    else:
        assert len(axes) == 3, "axes must be a list of 3 axes"
        assert axes is not None and fig is not None, "fig and axes must be provided if not None"

    axis_TSS_ = tuple(axis_ref_set.union(axis_set))
    axis_TSS_ = axis_TSS_ if axis_TSS_[0] is not None else None
    RSS_ = RSS.squeeze(axis=axis)
    TSS_ = TSS.squeeze(axis=axis_TSS_) # Collapse the axis_ref and axis dimensions

    im = plot_dimensional_score(RSS_, ax=axes[0])
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title('RSS')

    im = plot_dimensional_score(TSS_, ax=axes[1])
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title('TSS')

    score_ = score
    im = plot_dimensional_score(score, ax=axes[2], vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[2])
    axes[2].set_title('Dim-R2')

    if axislabels is not None:
        axis_all = list(range(y_true.ndim))
        axis_RSS = axis_all.copy()
        [axis_RSS.remove(ax) for ax in list(axis_set)]
        labels_RSS = np.array(axislabels)[axis_RSS]

        axis_TSS = axis_all.copy()
        [axis_TSS.remove(ax) for ax in axis_TSS_]
        labels_TSS = np.array(axislabels)[axis_TSS]

        if len(labels_RSS)==2:
            axes[0].set_xlabel(labels_RSS[1])
            axes[0].set_ylabel(labels_RSS[0])
            axes[2].set_xlabel(labels_RSS[1])
            axes[2].set_ylabel(labels_RSS[0])
        elif len(labels_RSS)==1:
            axes[0].set_xlabel(labels_RSS[0])
            axes[2].set_xlabel(labels_RSS[0])

        if len(labels_TSS)==2:
            axes[1].set_xlabel(labels_TSS[1])
            axes[1].set_ylabel(labels_TSS[0])
        elif len(labels_TSS)==1:
            axes[1].set_xlabel(labels_TSS[0])
    
    fig.tight_layout()
    return score, fig, axes

def axis_fix(axis, ndim):
    """
    Convert axis into a tuple of positive integers.
    
    Parameters
    ----------
    axis : int or iterable of int
        Axis or axes to normalize.
    
    ndim : int
        The maximum number of dimensions axis can have.
        Used as reference for converting negative indices.
    
    Returns
    -------
    axis : tuple of int
        Normalized axis as a tuple of non-negative integers.
    """
    if not isinstance(axis, Iterable): # axis is a single int
        assert type(axis) is int, 'type(axis) should be integer'
        axis = (axis,)
    
    axis = tuple(ax if ax>=0 else ndim+ax for ax in axis)
    return axis

def TSS_score(y_true, axis=None, axis_norm=None, axis_pool=None):

    # Axis fixing
    axis = tuple(range(y_true.ndim)) if axis is None else axis # Default to collapsing all dimensions
    axis_norm = axis if axis_norm is None else axis_norm
    axis_pool = axis_norm if axis_pool is None else axis_pool
    axis, axis_norm, axis_pool = axis_fix(axis, y_true.ndim), axis_fix(axis_norm, y_true.ndim), axis_fix(axis_pool, y_true.ndim)

    # axis trimming operations for computing TSS
    axis_set, axis_norm_set, axis_pool_set = set(axis), set(axis_norm), set(axis_pool)
    # axis_pool_set = {axis_pool} if not isinstance(axis_pool, Iterable) else set(axis_pool)
    # axis_set = set(axis)
    # axis_norm_set = {axis_norm} if axis_norm is not None and not isinstance(axis_norm, Iterable) else set(axis_norm) if axis_norm is not None else set()

    assert axis_norm_set.issubset(axis_pool_set), f'axis_norm {axis_norm} must be a subset of axis_pool {axis_pool} because axis_norm defines variability and axis_pool additionally averages'

    # Dimensions of TSS must be smaller than RSS, so mean/sum over axis_pool and axis
    axis_sum = axis
    axis_mean = tuple(axis_pool_set - axis_set) # Average over axis_pool - axis

    if not axis_set.issubset(axis_pool_set): # If axis is not a subset of axis_pool, expand axis_pool to include axis
        warnings.warn(f"axis {axis} is not a subset of axis_pool {axis_pool}, TSS sums over axis {axis_sum} and averages over remaining axis_pool {axis_mean}")

    # axis_norm used to compute the mean of y_true
    y_mean = np.mean(y_true, axis=axis_norm, keepdims=True)
    TS = (y_true - y_mean)**2 # Total Square (TS)

    # axis_pool used to aggregate additional dimensions (average) additional to axis
    TSS = np.mean(TS, axis=axis_mean, keepdims=True)
    TSS = np.sum(TSS, axis=axis_sum, keepdims=True)

    return TSS

def r2_score(y_true, y_pred, axis=None, axis_norm=None, axis_pool=None, force_finite=True, TSS=None):
    """
    R^2 score for multidimensional predictions.
    collapses all axes except the specified axis.

    Computes 1 - RSS / TSS, where RSS is the residual sum of squares and TSS is the total sum of squares.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    axis: int or iterable of int, default=None
        Axis to collapse.
        If None, collapses all axes to yield a single number.

            
    axis_norm: int or iterable of int, default=None
        axis used to measure y_true.mean(axis=axis_norm), to measure reference variability (TSS) for normalizing.
        It is recommended to keep axis_norm minimal, since a smaller axis_norm yields more localized TSS and a more detailed Dim-R2. 

    axis_pool: int or iterable of int, default=None
        axis used to additionally average TSS across for broader evaluation.
        It is recommended to keep axis_pool minimal, since a smaller axis_pool yields more localized normalization references, providing a more detailed Dim-R2.
    
    Returns
    -------
    z : np.ndarray
        if axis is specified, returns an array of shape with remaining axes.
        if axis=None, a single number is returned.
    """

    # Axis fixing
    axis = tuple(range(y_true.ndim)) if axis is None else axis # Default to collapsing all dimensions
    axis_norm = axis if axis_norm is None else axis_norm
    axis_pool = axis_norm if axis_pool is None else axis_pool
    axis, axis_norm, axis_pool = axis_fix(axis, y_true.ndim), axis_fix(axis_norm, y_true.ndim), axis_fix(axis_pool, y_true.ndim)

    # Residual Sum of Squares (RSS) 
    RS = (y_true - y_pred)**2 # Residual Square (RS)
    RSS = np.sum(RS, axis=axis, keepdims=True)

    # Total Sum of Squares (TSS)
    if TSS is None:
        TSS = TSS_score(y_true=y_true, axis=axis, axis_norm=axis_norm, axis_pool=axis_pool)
    else: # TSS is given
        try:
            TSS = np.broadcast_to(TSS, RSS.shape)
        except ValueError as e:
            raise ValueError(f'The shape of given TSS ({TSS.shape}) must be broadcastable to shape of RSS ({RSS.shape})') from e

    # R2
    score = 1 - RSS / TSS
    score = np.squeeze(score, axis=axis) # Collapse the axis dimension

    # if nan (TSS=0, y_true no variance) or -inf (RSS/TSS=inf, very bad prediction)
    if force_finite:
        score[np.isnan(score)] = 1
        score[np.isinf(score)] = 0 # -Inf means no fit, so set to 0

    # if len(axis) == y_true.ndim:
    if score.ndim==0: # score.ndim==0 is better than len(axis) == y_true.ndim?
        score = score.item()

    return score

def pearson_corrcoef(x, y, multioutput='uniform_average'):
    """
    Pearson correlation coefficient, expanded into 2D inputs with averaging

    Parameters
    ----------
    x : array-like of shape (n_samples,) or (n_samples, n_channels)
    y : array-like of shape (n_samples,) or (n_samples, n_channels)
    multioutput : {'uniform_average', 'variance_weighted', 'raw_values'} or array-like of weights
        How to aggregate per-channel correlations.
        - 'raw_values' -> return array of correlations per channel
        - 'uniform_average' -> return mean correlation (scalar)
        - array-like -> treated as weights for a weighted average

    Returns
    -------
    corr : float or ndarray
        Pearson correlation(s). If aggregation requested, returns scalar; otherwise,
        returns a 1d array of length n_channels.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape; got {x.shape} vs {y.shape}")

    # compute per-channel Pearson correlation
    corr = np.array([np.corrcoef(x_, y_, rowvar=False)[0,1] for x_, y_ in zip(x.T, y.T)])
    # corr = np.array([np.corrcoef(x_, y_, rowvar=True) for x_, y_ in zip(x.T, y.T)])

    # aggregation / multioutput handling
    if multioutput == 'raw_values':
        return corr
    elif multioutput == 'uniform_average':
        return corr.mean().item()
    else:
        raise ValueError(f"Unknown multioutput string: {multioutput}")

def corrcoef(x, y): # For sanity check with np.corrcoef()[0,1]
    return ((x*y).mean() - x.mean()*y.mean())/(x.std()*y.std())

# %%
'''
Signal generation functions
'''
def make_signals(N, C, stagger_phase=True):
    t = np.linspace(0, 2*np.pi, N)
    if stagger_phase:
        y_true = np.stack([np.sin(t+d) for d in np.linspace(0, 2*np.pi, C, endpoint=False)], axis=0) # (trials, time)
    else:
        y_true = np.stack([np.sin(t)]*C, axis=0)
    y_pred = y_true.copy()

    y_true, y_pred = y_true.T, y_pred.T
    return t, y_true, y_pred

def noise(y, noisescale='linear_increasing', noise='uniform'):
    """
    Add noise to the signal
    """
    if noisescale == 'linear_increasing':
        scale = np.linspace(0, 1, y.shape[0])
    elif noisescale == 'linear_decreasing':
        scale = np.linspace(1, 0, y.shape[0])
    elif noisescale == 'constant':
        scale = np.ones(y.shape[0])
    else:
        raise ValueError(f"Unknown noisescale: {noisescale}")
    
    if noise=='uniform': # uniform random between [-1,1]
        noise = (np.random.rand(*y.shape)*2 - 1)
    elif noise=='gaussian':
        noise = np.random.randn(*y.shape)
    else:
        raise ValueError(f"Unknown noise: {noise}")

    return scale * noise

# %%
'''
Plot
'''
def plot_dimensional_score(score, ax, vmin=None, vmax=None, cmap='Blues'):
    score_is_0d = isinstance(score, float) or (isinstance(score, np.ndarray) and score.ndim==0)
    score_is_1d = isinstance(score, np.ndarray) and score.ndim == 1
    if score_is_0d:
        score_plot = np.array([[score]])
    elif score_is_1d:
        score_plot = score.reshape(1, -1)
    else:
        score_plot = score
    im = ax.matshow(score_plot, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    if score_is_0d:
        ax.set_xticks([])
        ax.set_yticks([])
    elif score_is_1d:
        ax.set_yticks([])  # Hide y-ticks if score is 1D
        ax.xaxis.tick_bottom()
    else:
        ax.xaxis.tick_bottom()

    return im

def plot_individual(t, y_true, y_pred, old_r2, dim_r2, r2_per_channel, ylim=None, figsize=None):
    cmap = plt.get_cmap('tab10')
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
