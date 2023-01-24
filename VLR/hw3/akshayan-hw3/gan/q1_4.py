import os

import torch
import torch.nn.functional as F

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.4.1: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    loss_fn = torch.nn.MSELoss()
    a = torch.ones_like(discrim_real).cuda()
    b = torch.zeros_like(discrim_fake).cuda()
    real_loss = loss_fn(discrim_real, a)
    fake_loss = loss_fn(discrim_fake, b)
    return real_loss + fake_loss

def compute_generator_loss(discrim_fake):
    # TODO 1.4.1: Implement LSGAN loss for generator.
    loss_fn = torch.nn.MSELoss()
    c = torch.ones_like(discrim_fake).cuda()
    fake_loss = loss_fn(discrim_fake, c)
    return fake_loss


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.4.2: Run this line of code.
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
