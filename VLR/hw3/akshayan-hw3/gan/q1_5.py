import os
from pickle import TRUE
from venv import create
from numpy import gradient

import torch
from networks_without_jit import Discriminator, Generator
#from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    # loss = max_D E[D(real_data)] - E[D(fake_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    discrim_real = discrim_real.reshape(-1)
    discrim_fake = discrim_fake.reshape(-1)
    loss_dis = -torch.mean(discrim_real) + torch.mean(discrim_fake)
    gradient_p = torch.autograd.grad(discrim_interp, interp, torch.ones_like(discrim_interp, 
                                     dtype=torch.float32).cuda(), create_graph=True, retain_graph=True)[0]
    #print(gradient_p.shape) 
    gradient_p = torch.reshape(gradient_p, (gradient_p.shape[0], -1))
    #print(gradient_p.shape)
    norm_gradient_p = gradient_p.norm(2, dim=1)
    gp = torch.mean(torch.pow(norm_gradient_p-1, 2))
    #print("gp", gp.item(), "loss_dis", loss_dis.item())
    return loss_dis + lamb * gp


def compute_generator_loss(discrim_fake):
    # TODO 1.5.1: Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    discrim_fake = discrim_fake.reshape(-1)
    loss = -torch.mean(discrim_fake)
    return loss

if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.5.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=192,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
    )
