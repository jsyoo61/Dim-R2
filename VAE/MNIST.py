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
import vae_utils as U2

# %%
# os.chdir('VAE')

# %%
def MNIST(root):
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())

    x, y = train_dataset[0]
    input_shape = x.shape
    train_data = train_dataset.data.unsqueeze(1)/255
    test_data = test_dataset.data.unsqueeze(1)/255

    dataset_kwargs = {'get_f': None, 'grayscale': True, 'targets_type': 'regression'}
    train_dataset = U2.ImageDataset(data=train_data, targets=np.array(train_dataset.targets), **dataset_kwargs)
    test_dataset = U2.ImageDataset(data=test_data, targets=np.array(test_dataset.targets), **dataset_kwargs)

    data = {
    'info': {
        'n_classes': 10,
        'in_channels': 1,
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
path_save = T.Path('MNIST')
path_save.mkdir(exist_ok=True)

# %%
data = MNIST(path_save)
ds_train, ds_test = data['dataset']['train'], data['dataset']['test']
ds_train, ds_val = T.sklearn.model_selection.stratified_train_test_split_data(data=ds_train, y=ds_train.targets, random_state=cfg.runtime.random.seed)

# %%
network = U2.CNNVAE(n_channels=data['info']['in_channels'], n_filters=cfg.nn.n_filters, latent_dim=cfg.nn.latent_dim, EncoderClass=U2.CNNEncoder, DecoderClass=U2.DCNNDecoder2828)
network.to(device)

# %%
criterion = U2.Loss_VAE()

# %%
val_metrics = {
    'mean_absolute_error': U2.mean_absolute_error 
}

# %%
cfg_train = OmegaConf.to_container(cfg.train.updat, resolve=True)
cfg_val = OmegaConf.to_container(cfg.train.validation, resolve=True)
cfg_val['forward_f'] = U2.forward_vae

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
forward_f=U2.forward_vae

# %%
'''
Take a few snapshots of the model's predictions
'''
n_updates = 100
n_snapshots = 16
# results = hs.E.forward_network(network=network, dataset=ds_test, forward_f=U2.forward_vae)
# fig, axes = U2.plot_overview(results, figsize=cfg.plot.figsize, targets_type='category')
fig, axes, d_figaxes = U2.plot_snapshot(network, ds_test, forward_f=U2.forward_vae, figsize=cfg.plot.figsize)
fig.savefig(path_save/f'MNIST_{trainer.iter}.png', dpi=300)
for target, (fig, axes) in d_figaxes.items():
    fig.savefig(path_save/f'MNIST_{trainer.iter}_{target}.png', dpi=300)
    
for i in range(n_snapshots):
    trainer.train(step=n_updates)
    # results = hs.E.forward_network(network=network, dataset=ds_test, forward_f=U2.forward_vae)
    # fig, axes = U2.plot_overview(results, figsize=cfg.plot.figsize, targets_type='category')
    fig, axes, d_figaxes = U2.plot_snapshot(network, ds_test, forward_f=U2.forward_vae, figsize=cfg.plot.figsize)
    fig.savefig(path_save/f'MNIST_{trainer.iter}.png', dpi=300)
    for target, (fig, axes) in d_figaxes.items():
        fig.savefig(path_save/f'MNIST_{trainer.iter}_{target}.png', dpi=300)

# %%
trainer.train()
trainer.load_best_model()
trainer.save()

# %%
fig, axes, d_figaxes = U2.plot_snapshot(network, ds_test, forward_f=U2.forward_vae, figsize=cfg.plot.figsize)
fig.savefig(path_save/f'MNIST_{trainer.iter}.png', dpi=300)
for target, (fig, axes) in d_figaxes.items():
    fig.savefig(path_save/f'MNIST_{trainer.iter}_{target}.png', dpi=300)
    
# %%