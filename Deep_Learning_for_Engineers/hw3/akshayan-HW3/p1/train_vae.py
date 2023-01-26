'''
train and test VAE model on airfoils
'''

from ast import arg
from asyncore import write
from sklearn.metrics import label_ranking_average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from dataset import AirfoilDataset
from vae import VAE
from utils import *
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser(description="Airfoil generation using VAE")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    writer = SummaryWriter()

    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # define dataset and dataloader
    dataset = AirfoilDataset()
    airfoil_x = dataset.get_x()
    airfoil_dim = airfoil_x.shape[0]
    airfoil_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # hyperparameters
    latent_dim = 16 # please do not change latent dimension
    lr = args.lr    # learning rate
    num_epochs = args.epochs
    alpha = 0.005

    # build the model
    vae = VAE(airfoil_dim=airfoil_dim, latent_dim=latent_dim, hidden_dims_decoder=[32, 64, 128],
              hidden_dims_encoder=[128, 64, 32]).to(device)
    print("VAE model:\n", vae)

    # define your loss function here
    loss_fn1 = nn.MSELoss(reduction='sum')
    
    # define optimizer for discriminator and generator separately
    optim = Adam(vae.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.4)

    # train the VAE model
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_len = 0
        #loop = tqdm(enumerate(airfoil_dataloader), total=len(airfoil_dataloader), leave=False)
        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):
            optim.zero_grad()
            y_real = local_batch.to(device)
            out, mean, logvar = vae(y_real)
            # train VAE
            #print("mean: ", mean)
            # calculate customized VAE loss
            loss1 = loss_fn1(out, y_real)
            #print(logvar.shape, logvar.exp().shape, torch.pow(mean, 2).shape)
            loss2 = -0.5 * torch.sum((torch.ones_like(logvar) + logvar - logvar.exp() - torch.pow(mean, 2)))
            loss = loss1 + alpha*loss2
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            epoch_len += local_batch.shape[0]

            # print loss while training
            if (n_batch * args.batch_size + 1) % 30 != 0:
                #loop.set_description(f" Epoch [{epoch}/{num_epochs}]")
                #loop.set_postfix(loss=loss.item())
                writer.add_scalar("loss", loss.item(), epoch * len(airfoil_dataloader) + n_batch)
                print("Epoch: [{}/{}], Batch: {}, loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))

        lr_scheduler.step()
        writer.add_scalar("Epoch_loss", epoch_loss/epoch_len, epoch)
        print("Epoch ", epoch, ", loss ", epoch_loss/epoch_len)
    torch.save(vae.state_dict(), "vae.pth")
    # test trained VAE model
    num_samples = 100
    vae.eval()
    
    # reconstuct airfoils
    dataset = AirfoilDataset()
    real_airfoils = dataset.get_y()[:num_samples]
    recon_airfoils, __, __ = vae(torch.from_numpy(real_airfoils).to(device))
    
    if 'cuda' in device:
        recon_airfoils = recon_airfoils.detach().cpu().numpy()
    else:
        recon_airfoils = recon_airfoils.detach().numpy()

    # randomly synthesize airfoils
    noise = torch.randn((num_samples, latent_dim)).to(device)   # create random noise 
    gen_airfoils = vae.decode(noise)
    if 'cuda' in device:
        gen_airfoils = gen_airfoils.detach().cpu().numpy()
    else:
        gen_airfoils = gen_airfoils.detach().numpy()

    # plot real/reconstructed/synthesized airfoils
    plot_airfoils(airfoil_x, real_airfoils , "real_airfoils.png")
    plot_airfoils(airfoil_x, recon_airfoils, "recon_airfoils.png")
    plot_airfoils(airfoil_x, gen_airfoils, "gen_airfoils.png")
    

if __name__ == "__main__":
    main()