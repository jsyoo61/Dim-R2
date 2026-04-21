# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils as U
import tools as T
import tools.plot

# %%
time = 100
C = 5

t, y_true, y_pred = U.make_signals(time, C, stagger_phase=True)
bias_consistent = np.broadcast_to(np.arange(y_true.shape[-1]), y_true.shape)

y_true = y_true + bias_consistent
y_pred = y_true
y_mean = np.broadcast_to(y_true.mean(0), y_true.shape)

y_all = np.stack([y_true,y_pred,y_mean], axis=0)

# %%
# cmap='RdBu'
# cmap='Greys'
cmap='viridis'
vmin = y_all.min()
vmax = y_all.max()

fig, axes = plt.subplots(ncols=3, figsize=(15,5))
im = U.plot_dimensional_score(y_true, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax)
axes[0].set_title('$y$')
axes[0].set_ylabel('Time')
axes[0].set_xlabel('Channel')

im = U.plot_dimensional_score(y_pred, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax)
axes[1].set_title('$\hat{y}$')
axes[1].set_xlabel('Channel')
axes[1].yaxis.set_visible(False)

im = U.plot_dimensional_score(y_mean, ax=axes[2], cmap=cmap, vmin=vmin, vmax=vmax)
axes[2].set_title('$\\bar{y}$')
axes[2].set_xlabel('Channel')
axes[2].yaxis.set_visible(False)
T.plot.add_colorbar(axes[2])
fig.savefig('example.png', dpi=300)

# %%
list_axis_norm = [0,1,(0,1)]
d_y_pred = {}
d_y_mean = {}
for axis_norm in list_axis_norm:
    d_y_pred[axis_norm] = U.r2_score(y_true, y_pred, axis_norm=axis_norm)
    d_y_mean[axis_norm] = U.r2_score(y_true, y_mean, axis_norm=axis_norm)

# %%
df = pd.DataFrame([d_y_pred, d_y_mean], index=['$\hat{y}$', '$\\bar{y}$']).T
df.to_latex()