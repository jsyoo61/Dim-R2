# %%
import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch
import torchvision
import torchvision.transforms as transforms

import hideandseek as hs
import tools as T
import tools.sklearn

import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

# %%
# root='CelebA'
# os.chdir('VAE')
import vae_utils as U2

# %%
def CelebA(root):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)) # Original size: (218,178)
    ])

    train_dataset = torchvision.datasets.CelebA(root=root, split='train', target_type='attr', download=True, transform=transform)
    test_dataset = torchvision.datasets.CelebA(root=root, split='test', target_type='attr', download=True, transform=transform)

    x, y = train_dataset[0]
    input_shape = x.shape # (3, 128, 128)
    in_channels = x.shape[0]
    n_classes = len(y)

    train_dataset = U2.DictWrapperDataset(train_dataset)
    test_dataset = U2.DictWrapperDataset(test_dataset)

    data = {
    'info': {
        'n_classes': n_classes,
        'in_channels': in_channels,
        'input_shape': input_shape
        },
    'dataset': {
        'train': train_dataset,
        'test': test_dataset
        },
    }
    return data

# %%
cfg = OmegaConf.load('cfg.yaml')
print(OmegaConf.to_yaml(cfg))

T.torch.seed(cfg.runtime.random.seed, strict=cfg.runtime.random.strict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
path_save = T.Path('CelebA')
path_save.mkdir(exist_ok=True)

# %%
root='C:\\Users\\jsyoo61\\Documents\\CelebA'
os.listdir(root)

data = CelebA(root)
ds_train, ds_test = data['dataset']['train'], data['dataset']['test']
ds_train, ds_val = T.sklearn.model_selection.train_test_split_data(data=ds_train, y=np.empty(len(ds_train)), random_state=cfg.runtime.random.seed, test_size=0.1)

# %%
network = U2.CNNVAE(n_channels=data['info']['in_channels'], n_filters=cfg.nn.n_filters, latent_dim=cfg.nn.latent_dim, EncoderClass=U2.CNNEncoder, DecoderClass=U2.DCNNDecoder128128)
network.to(device)

# %%
import torch.nn as nn
criterion = U2.Loss_VAE(loss_recon=nn.BCEWithLogitsLoss())

# %%
val_metrics = {
    'mean_absolute_error': U2.mean_absolute_error 
}

# %%
cfg_train = OmegaConf.to_container(cfg.train.updat, resolve=True)
cfg_val = OmegaConf.to_container(cfg.train.validation, resolve=True)
cfg_val['forward_f'] = U2.forward_vae_with_logits
# cfg_val['forward_f'] = U2.forward_vae

kwargs = {
    'network': network,
    'network_dir': path_save/'network',
    'train_dataset': ds_train,
    'cfg_train': cfg_train,
    'criterion': criterion,
    'val_dataset': ds_val,
    'cfg_val': cfg_val,
    'val_metrics': val_metrics,
}

trainer = U2.VAETrainer(**kwargs)

# %%
n_attr_cols=8
def trim_attr(results, n_attr_cols):
    # Choose n most even attribute columns
    y = results['y']
    attr_cols = np.argsort(np.abs(y.sum(0)/len(y)-0.5))[:n_attr_cols]
    attr_cols_name = [ds_train.dataset.attr_names[i] for i in attr_cols]
    results['y'] = y[:,attr_cols]
    return results, attr_cols, attr_cols_name

# %%
'''
Take a few snapshots of the model's predictions
'''
n_updates = 100
n_snapshots = 16
import matplotlib.pyplot as plt

# results = hs.E.forward_network(network=network, dataset=ds_test, forward_f=U2.forward_vae, batch_size=cfg.train.validation.batch_size)
results = hs.E.forward_network(network=network, dataset=ds_test, forward_f=U2.forward_vae_with_logits, batch_size=cfg.train.validation.batch_size)
results, attr_cols, attr_cols_name = trim_attr(results, n_attr_cols=8)

fig, axes, d_figaxes = U2.plot_overview(results, figsize=cfg.plot.figsize, targets_type='onehot')
[ax.set_ylabel(attr_col_name) for ax, attr_col_name in zip(axes[:,0], attr_cols_name)]
fig.savefig(path_save/f'CelebA_{trainer.iter}.png', dpi=300)

for target, (fig, axes) in d_figaxes.items():
    attr_col_name = attr_cols_name[target]
    axes[0].set_ylabel(attr_col_name)
    fig.savefig(path_save/f'CelebA_{trainer.iter}_{attr_col_name}.png', dpi=300)

plt.close(fig)
[plt.close(fig) for (fig, axes) in d_figaxes.values()]
# fig, axes = U2.plot_snapshot(network, ds_test, forward_f=U2.forward_vae, figsize=cfg.plot.figsize, targets_type='onehot')
for i in range(n_snapshots):
    trainer.train(step=n_updates)
    # results = hs.E.forward_network(network=network, dataset=ds_test, forward_f=U2.forward_vae, batch_size=cfg.train.validation.batch_size)
    results = hs.E.forward_network(network=network, dataset=ds_test, forward_f=U2.forward_vae_with_logits, batch_size=cfg.train.validation.batch_size)
    results['y'] = results['y'][:,attr_cols]
    fig, axes, d_figaxes = U2.plot_overview(results, figsize=cfg.plot.figsize, targets_type='onehot')
    [ax.set_ylabel(attr_col_name) for ax, attr_col_name in zip(axes[:,0], attr_cols_name)]
    fig.savefig(path_save/f'CelebA_{trainer.iter}.png', dpi=300)

    for target, (fig, axes) in d_figaxes.items():
        attr_col_name = attr_cols_name[target]
        axes[0].set_ylabel(attr_col_name)
        fig.savefig(path_save/f'CelebA_{trainer.iter}_{attr_col_name}.png', dpi=300)

    
    plt.close(fig)
    [plt.close(fig) for (fig, axes) in d_figaxes.values()]

# %%
trainer.train()
trainer.load_best_model()
trainer.save(path_save)

# %%
results = hs.E.forward_network(network=network, dataset=ds_test, forward_f=U2.forward_vae_with_logits, batch_size=cfg.train.validation.batch_size)
results['y'] = results['y'][:,attr_cols]
fig, axes, d_figaxes = U2.plot_overview(results, figsize=cfg.plot.figsize, targets_type='onehot')
[ax.set_ylabel(attr_col_name) for ax, attr_col_name in zip(axes[:,0], attr_cols_name)]
fig.savefig(path_save/f'CelebA_{trainer.iter}.png', dpi=300)

for target, (fig, axes) in d_figaxes.items():
    attr_col_name = attr_cols_name[target]
    axes[0].set_ylabel(attr_col_name)
    fig.savefig(path_save/f'CelebA_{trainer.iter}_{attr_col_name}.png', dpi=300)

plt.close(fig)
[plt.close(fig) for (fig, axes) in d_figaxes.values()]

# %%
trainer.iter
dir(trainer)
dir(trainer.earlystopper)
trainer.earlystopper.best_network
trainer.earlystopper.discard_best

# %%
forward_f = U2.forward_vae
dataset = ds_test
batch_size=cfg.train.validation.batch_size
targets_type=None
num_workers=0
amp=False
from functools import partial
import torch.utils.data as D
ds_test[0]
import matplotlib.pyplot as plt
figsize=cfg.plot.figsize
ds_test.dataset[0]
targets_type='onehot'


# %%
y = results['y']
np.unique(y)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
cluster = KMeans(n_clusters=5)
y_label = cluster.fit_predict(y)
np.unique(y_label)

dir(cluster)
np.round(cluster.cluster_centers_)

pca = PCA(n_components=5)
y_distrib = pca.fit_transform(y)

dir(ds_test)
list(ds_test.dataset.attr_names)
len(ds_test.dataset.attr_names)

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for label in np.unique(y_label):
    i = y_label==label
    ax.scatter(*y_distrib[i,:2].T, label=label)
ax.legend()

# %%
# !pip install tools --upgrade 
# !pip install hideandseek --upgrade 