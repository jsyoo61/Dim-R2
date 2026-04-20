# %%
import itertools as it
import warnings

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import hideandseek as hs
import tools as T
import tools.plot

import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import utils as U

# %%
'''
Dataset
'''
def infer_targets_type(targets):
    classes = sorted(list(set(np.array(targets))))
    if set(classes)==set([0,1]):
        return 'binary'
    else:
        return 'categorical'

def infer_targets_dtype(targets_type):
    if targets_type == 'binary':
        return torch.float32
    elif targets_type == 'categorical':
        return torch.long
    
class ImageDataset(D.Dataset):
    '''
    must specify get_f if you want to load directly from drive.
    '''
    def __init__(self, data, targets, get_f=None, grayscale=False, targets_type=None, targets_dtype=None):
        self.data = data
        self.targets = targets
        self.get_f = get_f
        self.grayscale = grayscale

        if targets_type is None:
            self.targets_type = infer_targets_type(self.targets)
        else:
            self.targets_type = targets_type
        if targets_dtype is None:
            self.targets_dtype = infer_targets_dtype(self.targets_type)
        else:
            self.targets_dtype = targets_dtype

    def __getitem__(self, idx):
        # Resize to 224?
        x = self.get_f(self.data[idx]) if self.get_f is not None else self.data[idx] # if get_f is none, it will most likely raise an error
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(self.targets[idx], dtype=self.targets_dtype)
        return {'x':x, 'y':y}

    def __len__(self):
        return len(self.targets)

class DictWrapperDataset(D.Dataset):
    attr_list = ['dataset', 'keys', 'keys_single']
    def __init__(self, dataset, keys=None):
        self.dataset = dataset

        sample = dataset[0]
        if keys is None:
            warnings.warn('Inferring keys from dataset sample may result in undesired variable keys. Please provide your own keys argument.')
            if not isinstance(sample, torch.Tensor):
                sample_keys = it.chain([chr(letter_code) for letter_code in range(120,123)], [chr(letter_code) for letter_code in range(97,120)])
                sample_keys = list(sample_keys)

                assert len(sample)<=len(sample_keys), f'Too many arguments ({len(sample)}) returned from dataset, please provide your own key'
                keys = [key for key, arg in zip(sample_keys, sample)]
                keys
            else:
                keys = ['x']
        else:
            assert len(keys)==len(sample)

        self.keys = keys
        self.keys_single = len(keys)==1

        # # Copy attributes from the original dataset
        # keys = [key for key in dir(dataset) if key.startswith('__') is False]
        # for key in keys:
        #     setattr(self, key, getattr(dataset, key))
        duplicate_keys = [key for key in dir(self.dataset) if key in self.attr_list]
        if len(duplicate_keys)>0:
            warnings.warn(f'The following attributes from dataset will not be accessible by getattr as it overlaps with ProxyDataset\'s attributes: {duplicate_keys}')

    def __repr__(self):
        return f'DictWrapperDataset({self.dataset})'
    
    def __getitem__(self, idx):
        if self.keys_single:
            return {self.keys[0]: self.dataset[idx]}
        else:
            return {k: v for k, v in zip(self.keys, self.dataset[idx])}

    def __getattr__(self, key):
        if key in self.attr_list:
            return getattr(self, key)
        else:
            return getattr(self.dataset, key)

    def __len__(self):
        return len(self.dataset)

# %%
'''
Neural Network
'''
# Start by first defining a convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class CNNEncoder(nn.Module):
    def __init__(self, in_channels, n_filters, n_outputs):
        super().__init__()

        self.in_channels=in_channels
        self.n_filters=n_filters
        self.n_outputs=n_outputs

        self.layers = nn.Sequential(
        ConvBlock(self.in_channels, self.n_filters, kernel_size=5, stride=2, padding=2),
        ConvBlock(self.n_filters, 2*self.n_filters, kernel_size=5, stride=2, padding=2),
        ConvBlock(2*self.n_filters, 4*self.n_filters, kernel_size=3, stride=2, padding=1),
        ConvBlock(4*self.n_filters, 6*self.n_filters, kernel_size=3, stride=2, padding=1),
        nn.Flatten(),
        nn.LazyLinear(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, self.n_outputs*2),
        )

    def forward(self, x):
        z = self.layers(x)
        z_mu, z_logsigma = z[...,:self.n_outputs], z[...,self.n_outputs:]
        return z_mu, z_logsigma

class DCNNDecoder(nn.Module):
    def __init__(self, n_inputs, n_filters, out_channels):
        super().__init__()

        self.n_inputs=n_inputs
        self.n_filters=n_filters
        self.out_channels=out_channels

    def forward(self, z):
        return self.layers(z)

class DCNNDecoder2828(DCNNDecoder):
    def __init__(self, n_inputs, n_filters, out_channels):
        super().__init__(n_inputs, n_filters, out_channels)

        # Linear (fully connected) layer to project from latent space
        self.layers = nn.Sequential(
            nn.LazyLinear(7 * 7 * 6 * self.n_filters),
            nn.ReLU(),
            nn.Unflatten(1, (6 * self.n_filters, 7, 7)),

            # Convolutional upsampling (inverse of an encoder)
            # [B, 6n_filters, 7, 7] -> [B, 4n_filters, 14, 14]
            nn.ConvTranspose2d(in_channels=6*self.n_filters,out_channels=4*self.n_filters,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            # [B, 4n_filters, 14, 14] -> [B, 2n_filters, 28, 28]
            nn.ConvTranspose2d(in_channels=4*self.n_filters,out_channels=1*self.out_channels,kernel_size=5,stride=2,padding=2,output_padding=1),
        )

class DCNNDecoder6464(DCNNDecoder):
    def __init__(self, n_inputs, n_filters, out_channels):
        super().__init__(n_inputs, n_filters, out_channels)

        # Linear (fully connected) layer to project from latent space
        self.layers = nn.Sequential(
            nn.LazyLinear(4 * 4 * 6 * self.n_filters),
            nn.ReLU(),
            nn.Unflatten(1, (6 * self.n_filters, 4, 4)),

            # Convolutional upsampling (inverse of an encoder)
            nn.ConvTranspose2d(in_channels=6*self.n_filters,out_channels=4*self.n_filters,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            # [B, 4n_filters, 8, 8] -> [B, 2n_filters, 16, 16]
            nn.ConvTranspose2d(in_channels=4*self.n_filters,out_channels=2*self.n_filters,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            # [B, 2n_filters, 16, 16] -> [B, self.n_filters, 32, 32]
            nn.ConvTranspose2d(in_channels=2*self.n_filters,out_channels=self.n_filters,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.ReLU(),
            # [B, self.n_filters, 32, 32] -> [B, self.out_channels, 64, 64]
            nn.ConvTranspose2d(in_channels=self.n_filters,out_channels=self.out_channels,kernel_size=5,stride=2,padding=2,output_padding=1),
        )

class DCNNDecoder128128(DCNNDecoder):
    def __init__(self, n_inputs, n_filters, out_channels):
        super().__init__(n_inputs, n_filters, out_channels)

        # Linear (fully connected) layer to project from latent space
        # to a 4 x 4 feature map with (6*n_filters) channels
        self.layers = nn.Sequential(
            nn.LazyLinear(4 * 4 * 6 * self.n_filters),
            nn.ReLU(),
            nn.Unflatten(1, (6 * self.n_filters, 4, 4)),

            # Convolutional upsampling (inverse of an encoder)
            nn.ConvTranspose2d(in_channels=6*self.n_filters,out_channels=4*self.n_filters,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            # [B, 4n_filters, 8, 8] -> [B, 2n_filters, 16, 16]
            nn.ConvTranspose2d(in_channels=4*self.n_filters,out_channels=2*self.n_filters,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            # [B, 2n_filters, 16, 16] -> [B, 2n_filters, 32, 32]
            nn.ConvTranspose2d(in_channels=2*self.n_filters,out_channels=2*self.n_filters,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.ReLU(),
            # [B, 2n_filters, 32, 32] -> [B, self.n_filters, 64, 64]
            nn.ConvTranspose2d(in_channels=2*self.n_filters,out_channels=self.n_filters,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.ReLU(),
            # [B, self.n_filters, 64, 64] -> [B, self.out_channels, 128, 128]
            nn.ConvTranspose2d(in_channels=self.n_filters,out_channels=self.out_channels,kernel_size=5,stride=2,padding=2,output_padding=1),
        )

class CNNVAE(nn.Module):
    def __init__(self, n_channels, n_filters=10, latent_dim=100, EncoderClass=CNNEncoder, DecoderClass=DCNNDecoder6464):
        super().__init__()
        self.n_channels=n_channels
        self.n_filters=n_filters
        self.latent_dim=latent_dim

        self.encoder = EncoderClass(in_channels=self.n_channels, n_filters=self.n_filters, n_outputs=self.latent_dim)
        self.decoder = DecoderClass(n_inputs=self.latent_dim, n_filters=self.n_filters, out_channels=self.n_channels)

    def forward(self, x, stochastic=False, return_z=False):
        z_mu, z_logsigma = self.encoder(x)
        z = sampling(z_mu, z_logsigma) if stochastic else z_mu
        x_hat = self.decoder(z)
        
        if return_z:
            return x_hat, z_mu, z_logsigma
        else:
            return x_hat

def sampling(z_mu, z_logsigma):
    eps = torch.randn_like(z_mu)
    z = z_mu + eps * torch.exp(z_logsigma)
    return z

# %%
'''
Loss
'''
class Loss_VAE(nn.Module):
    def __init__(self, kl_weight=0.0005, loss_recon=nn.L1Loss()):
        super().__init__()
        self.kl_weight=kl_weight
        self.loss_recon=loss_recon

    def forward(self, x, x_hat, z_mu, z_logsigma):
        latent_loss = (torch.exp(z_logsigma) + z_mu**2 - 1 - z_logsigma) / 2
        reconstruction_loss = self.loss_recon(x_hat, x)
        
        vae_loss = self.kl_weight * latent_loss.mean() + reconstruction_loss
        return vae_loss

# %%
'''
metrics
'''
def mean_absolute_error(x_true, x_hat):
    N = len(x_true)
    x_true, x_hat = x_true.reshape(N,-1), x_hat.reshape(N,-1)
    score = sk_metrics.mean_absolute_error(x_true, x_hat)
    return score

def forward_vae(network, x, y):
    x_hat = network(x, stochastic=False, return_z=False)

    result_dict = {
        'x_true': x,
        'y': y,
        'x_hat': x_hat,
    }
    return result_dict

def forward_vae_with_logits(network, x, y):
    x_logit = network(x, stochastic=False, return_z=False)
    x_hat = torch.sigmoid(x_logit)

    result_dict = {
        'x_true': x,
        'y': y,
        'x_hat': x_hat,
    }
    return result_dict
# %%
'''
Trainer
'''
class VAETrainer(hs.Trainer):
    def forward(self, x, y):
        # target label y not used
        x_hat, z_mu, z_logsigma = self.network(x, stochastic=True, return_z=True)
        loss = self.criterion(x, x_hat, z_mu, z_logsigma)
        return loss

# %%
'''
Plot
'''
def plot_overview(results, figsize=(5,5), targets_type='category'):
    # divide data per category
    x_true, x_hat, y = results['x_true'], results['x_hat'], results['y']
    assert x_true.ndim==4 # (Batch, channel, W, H)
    if x_true.shape[1]==1:
        greyscale=True
    elif x_true.shape[1]==3 or x_true.shape[1]==4:
        greyscale=False
    else:
        raise Exception(f'x_true has channels of {x_true.shape[1]}, which must be either one of [1, 3, 4]')

    d_data_targets = {}
    assert targets_type in ['category', 'onehot']

    if targets_type=='category':
        targets = np.unique(y)
        d_targets_i = {target: y==target for target in targets}
    elif targets_type=='onehot':
        assert y.ndim==2
        y = y.astype(bool) # (observation, targets)
        d_targets_i = {target: y_ for target, y_ in enumerate(y.T)}

    for target, i in d_targets_i.items():
        x_true_, x_hat_ = x_true[i], x_hat[i]
        d_data_targets[target] = {
            'x_true': x_true_,
            'x_hat':x_hat_,
        }

    # Plot
    ncols=4
    nrows=len(d_targets_i)
    cbar_width=0.2
    cmap='Blues' if greyscale else 'Greys_r'

    def plot_true_hat(d_data_temp, axes):
        assert len(axes.flatten())==4 or len(axes.flatten())==7
        x_true_mean, x_hat_mean, dim_r2 = d_data_temp['x_true_mean'], d_data_temp['x_hat_mean'], d_data_temp['dim-r2'] 

        axes[0].matshow(x_true_mean, cmap='Grays', vmin=0, vmax=1)
        axes[1].matshow(x_hat_mean, cmap='Grays', vmin=0, vmax=1)
        im = axes[2].matshow(dim_r2, cmap=cmap)
        cbar = T.plot.add_colorbar(axes[2], mappable=im, width=cbar_width, divide=True)
        im = axes[3].matshow(dim_r2, vmin=0, vmax=1, cmap=cmap)
        cbar = T.plot.add_colorbar(axes[3], mappable=im, width=cbar_width, divide=True)

        if not greyscale:
            x_true, x_hat = d_data_temp['x_true'], d_data_temp['x_hat']
            dim_r2_channels = np.transpose(U.r2_score(x_true, x_hat, axis=0, axis_norm=(2,3)), (1,2,0)) # (W, H, 3)
            im = axes[4].matshow(dim_r2_channels[...,0], vmin=0, vmax=1, cmap='Reds_r')
            cbar = T.plot.add_colorbar(axes[4], mappable=im, width=cbar_width, divide=True)
            im = axes[5].matshow(dim_r2_channels[...,1], vmin=0, vmax=1, cmap='Greens_r')
            cbar = T.plot.add_colorbar(axes[5], mappable=im, width=cbar_width, divide=True)
            im = axes[6].matshow(dim_r2_channels[...,2], vmin=0, vmax=1, cmap='Blues_r')
            cbar = T.plot.add_colorbar(axes[6], mappable=im, width=cbar_width, divide=True)

        [ax.set_xticklabels([]) for ax in axes]
        [ax.set_xticks([]) for ax in axes]
        [ax.set_yticks([]) for ax in axes]
        [ax.set_yticklabels([]) for ax in axes]

    def set_axes_title(axes):
        axes[0].set_title(f'$\\bar{{y}}$')
        axes[1].set_title(f'$\\bar{{\\hat{{y}}}}$')
        axes[2].set_title(f'$Dim R2$')
        axes[3].set_title(f'$Dim R2$')
        if not greyscale:
            axes[4].set_title('Dim R2 (Channel R)')
            axes[5].set_title('Dim R2 (Channel G)')
            axes[6].set_title('Dim R2 (Channel B)')

    if greyscale:
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    else:
        fig, axes = plt.subplots(ncols=ncols+3, nrows=nrows, figsize=(figsize[0]*ncols, figsize[1]*nrows))

    d_figaxes = {}

    for (target, d_data), axes_ in zip(d_data_targets.items(), axes):
        x_true, x_hat = d_data['x_true'], d_data['x_hat']
        x_true_mean = np.transpose(x_true.mean(0), (1,2,0)) # (W, H, Channel)
        x_hat_mean = np.transpose(x_hat.mean(0), (1,2,0)) # (W, H, Channel)
        dim_r2 = np.expand_dims(U.r2_score(x_true, x_hat, axis=(0,1), axis_norm=(2,3)),2) # (W, H, 1)

        d_data_temp = {
            'x_true_mean': x_true_mean,
            'x_hat_mean': x_hat_mean,
            'dim-r2': dim_r2,
            'x_true': x_true,
            'x_hat': x_hat,
        }
        # (Batch, Channel, W, H)
        plot_true_hat(d_data_temp, axes_)

        ncols_ind, nrows_ind=ncols, 1

        if greyscale:
            fig_ind, axes_ind = plt.subplots(ncols=ncols_ind, nrows=nrows_ind, figsize=(figsize[0]*ncols_ind, figsize[1]*nrows_ind))
        else:
            fig_ind, axes_ind = plt.subplots(ncols=ncols_ind+3, nrows=nrows_ind, figsize=(figsize[0]*ncols_ind, figsize[1]*nrows_ind))

        plot_true_hat(d_data_temp, axes_ind)
        set_axes_title(axes_ind)
        fig_ind.tight_layout()

        d_figaxes[target] = (fig_ind, axes_ind)

    set_axes_title(axes[0])
    fig.tight_layout()

    return fig, axes, d_figaxes

def plot_snapshot(network, ds_test, forward_f, figsize=(5,5), targets_type='category'):
    results = hs.E.forward_network(network=network, dataset=ds_test, forward_f=forward_f)
    print(results.keys())
    fig, axes, d_figaxes = plot_overview(results, figsize=figsize, targets_type=targets_type)
    # fig, axes, d_figaxes = plot_overview(results, figsize=cfg.plot.figsize)

    return fig, axes, d_figaxes

# %%
