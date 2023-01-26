'''
train and test GAN model on airfoils
'''

from email import generator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import argparse

from dataset import AirfoilDataset
from gan import Discriminator, Generator
from utils import *
from torch.utils.tensorboard import SummaryWriter

def main():
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description="Airfoil generation using GAN")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    writer = SummaryWriter()

    # define dataset and dataloader
    dataset = AirfoilDataset()
    airfoil_x = dataset.get_x()
    airfoil_dim = airfoil_x.shape[0]
    airfoil_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # hyperparameters
    latent_dim = 16 # please do not change latent dimension
    lr_dis = 0.0002 # discriminator learning rate
    lr_gen = 0.00015 # generator learning rate
    num_epochs = 60
    
    # build the model
    dis = Discriminator(input_dim=airfoil_dim, hidden_dims=[128, 64, 32]).to(device)
    gen = Generator(latent_dim=latent_dim, airfoil_dim=airfoil_dim, hidden_dims=[32, 64, 128]).to(device)
    print("Distrminator model:\n", dis)
    print("Generator model:\n", gen)

    # define your GAN loss function here
    # you may need to define your own GAN loss function/class
    loss_fn_dis = gan_loss_disc
    loss_fn_gen = gan_loss_gen

    # define optimizer for discriminator and generator separately
    optim_dis = Adam(dis.parameters(), lr=lr_dis)
    optim_gen = Adam(gen.parameters(), lr=lr_gen)
    
    # train the GAN model
    for epoch in range(args.epochs):
        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):
            y_real = local_batch.to(device)

            # train discriminator
            real_pred = dis(y_real)
            noise = torch.randn((args.batch_size, latent_dim)).to(device)
            y_fake = gen(noise)
            fake_pred = dis(y_fake)

            # calculate customized GAN loss for discriminator
            loss_dis = loss_fn_dis(real_pred, fake_pred)
            optim_dis.zero_grad()
            loss_dis.backward()
            optim_dis.step()

            # train generator
            # calculate customized GAN loss for generator
            noise = torch.randn((args.batch_size, latent_dim)).to(device)
            y_fake = gen(noise)
            fake_pred = dis(y_fake)
            loss_gen = loss_fn_gen(fake_pred)
            optim_gen.zero_grad()
            loss_gen.backward()
            optim_gen.step()

            # print loss while training
            if (n_batch + 1) % 10 != 0:
                print("Epoch: [{}/{}], Batch: {}, Discriminator loss: {}, Generator loss: {}".format(
                    epoch, args.epochs, n_batch, loss_dis.item(), loss_gen.item()))
                writer.add_scalar("Disc loss", loss_dis.item(), epoch * len(airfoil_dataloader) + n_batch)
                writer.add_scalar("Gen loss", loss_gen.item(), epoch * len(airfoil_dataloader) + n_batch)

    torch.save(gen.state_dict(), "gen.pth")
    torch.save(dis.state_dict(), "dis.pth")
    # test trained GAN model
    num_samples = 100
    # create random noise 
    noise = torch.randn((num_samples, latent_dim)).to(device)
    # generate airfoils
    gen_airfoils = gen(noise)
    if 'cuda' in device:
        gen_airfoils = gen_airfoils.detach().cpu().numpy()
    else:
        gen_airfoils = gen_airfoils.detach().numpy()

    # plot generated airfoils
    plot_airfoils(airfoil_x, gen_airfoils, "gen_airfoils_gan.png")


if __name__ == "__main__":
    main()

