from collections import OrderedDict
from operator import mod
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from model import AEModel
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import time
import os
from utils import *
from torch.utils.tensorboard import SummaryWriter
from wandb import wandb

wandb.init(project="hw3_AE")

def ae_loss(model, x):
    """ 
    TODO 2.1.2: fill in MSE loss between x and its reconstruction. 
    return loss, {recon_loss = loss} 
    """
    loss_fn = nn.MSELoss(reduction='sum')
    latent_vec = model.encoder(x)
    reconstruction = model.decoder(latent_vec)
    sum_loss = loss_fn(reconstruction, x)
    loss = sum_loss / x.shape[0]
    return loss, OrderedDict(recon_loss=loss)

def vae_loss(model, x, beta = 1):
    """TODO 2.2.2 : Fill in recon_loss and kl_loss. """

    loss_fn = nn.MSELoss(reduction='sum')
    mean, logvar = model.encoder(x)
    rand_number = torch.randn_like(logvar).cuda()
    sampled_no = mean + rand_number * torch.exp(0.5*logvar)
    output = model.decoder(sampled_no)
    recon_loss = loss_fn(output, x)
    recon_loss /= x.shape[0]
    kl_loss = 0.5 * torch.sum(-logvar - 1 + torch.exp(logvar) + torch.pow(mean, 2))
    kl_loss /= x.shape[0]
    total_loss = recon_loss + beta*kl_loss
    return total_loss, OrderedDict(recon_loss=recon_loss, kl_loss=kl_loss)


def constant_beta_scheduler(target_val = 1):
    def _helper(epoch):
        return target_val
    return _helper

def linear_beta_scheduler(max_epochs=None, target_val = 1):
    """TODO 2.3.2 : Fill in helper. The value returned should increase linearly 
    from 0 at epoch 0 to target_val at epoch max_epochs """
    def _helper(epoch):
       return (epoch + 1) * target_val / max_epochs
    return _helper

def run_train_epoch(model, loss_mode, train_loader, optimizer, beta = 1, grad_clip = 1):
    print("Beta: ", beta)
    model.train()
    all_metrics = []
    for x, _ in train_loader:
        x = preprocess_data(x)
        if loss_mode == 'ae':
            loss, _metric = ae_loss(model, x)
        elif loss_mode == 'vae':
            loss, _metric = vae_loss(model, x, beta)
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return avg_dict(all_metrics)


def get_val_metrics(model, loss_mode, val_loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = preprocess_data(x)
            if loss_mode == 'ae':
                _, _metric = ae_loss(model, x)
            elif loss_mode == 'vae':
                _, _metric = vae_loss(model, x)
            all_metrics.append(_metric)

    return avg_dict(all_metrics)

def main(log_dir, loss_mode = 'vae', beta_mode = 'constant', num_epochs = 20, batch_size = 256, latent_size = 256,
         target_beta_val = 1, grad_clip=1, lr = 1e-3, eval_interval = 5):

    writer = SummaryWriter(log_dir="runs/" + loss_mode)
    os.makedirs('data/'+ log_dir, exist_ok = True)
    train_loader, val_loader = get_dataloaders()

    variational = True if loss_mode == 'vae' else False
    model = AEModel(variational, latent_size, input_shape = (3, 32, 32)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    vis_x = next(iter(val_loader))[0][:36]
    
    #beta_mode is for part 2.3, you can ignore it for parts 2.1, 2.2
    if beta_mode == 'constant':
        beta_fn = constant_beta_scheduler(target_val = target_beta_val) 
    elif beta_mode == 'linear':
        beta_fn = linear_beta_scheduler(max_epochs=num_epochs, target_val = target_beta_val) 

    for epoch in range(num_epochs):
        print('epoch', epoch)
        train_metrics = run_train_epoch(model, loss_mode, train_loader, optimizer, beta_fn(epoch))
        val_metrics = get_val_metrics(model, loss_mode, val_loader)
        #print(val_metrics)
        #TODO : add plotting code for metrics (required for multiple parts)
        #writer.add_scalar("AE/val_reconstruction_loss", val_metrics['recon_loss'], epoch)
        
        wandb.log({ log_dir + " Reconstruction Loss": val_metrics['recon_loss'], "epoch": epoch})
        if loss_mode == 'vae':
            wandb.log({ log_dir + " KL Loss": val_metrics['kl_loss'], "epoch": epoch})

        if (epoch+1)%eval_interval == 0:
            print(epoch, train_metrics)
            print(epoch, val_metrics)

            vis_recons(model, vis_x, 'data/'+log_dir+ '/epoch_'+str(epoch))
            if loss_mode == 'vae':
                vis_samples(model, 'data/'+log_dir+ '/epoch_'+str(epoch) )


if __name__ == '__main__':
    #TODO: Experiments to run : 
    #2.1 - Auto-Encoder
    #Run for latent_sizes 16, 128 and 1024
    # main('ae_latent1024', loss_mode = 'ae',  num_epochs = 20, latent_size = 1024)
    # main('ae_latent128', loss_mode = 'ae',  num_epochs = 20, latent_size = 128)
    # main('ae_latent16', loss_mode = 'ae',  num_epochs = 20, latent_size = 16)

    #Q 2.2 - Variational Auto-Encoder
    # main('vae_latent1024', loss_mode = 'vae', num_epochs = 20, latent_size = 1024)
    # main('vae_latent128', loss_mode = 'vae', num_epochs = 20, latent_size = 128)
    # main('vae_latent16', loss_mode = 'vae', num_epochs = 20, latent_size = 16)

    #Q 2.3.1 - Beta-VAE (constant beta)
    #Run for beta values 0.8, 1.2
    # main('vae_latent1024_beta_constant0.8', loss_mode = 'vae', beta_mode = 'constant', 
    #      target_beta_val = 0.8, num_epochs = 20, latent_size = 1024)
    # main('vae_latent1024_beta_constant1.2', loss_mode = 'vae', beta_mode = 'constant', 
    #      target_beta_val = 1.2, num_epochs = 20, latent_size = 1024)

    #Q 2.3.2 - VAE with annealed beta (linear schedule)
    main('vae_latent1024_beta_linear1', loss_mode = 'vae', beta_mode = 'linear', target_beta_val = 1, num_epochs = 20, latent_size = 1024)