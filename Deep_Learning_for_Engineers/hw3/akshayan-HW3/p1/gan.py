import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super(Discriminator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, 1)
        # since discriminator is a binary classifier
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())

        if len(hidden_dims) != 0:
            layers.append(nn.Linear(hidden_dims[-1], 1))
        else:
            layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # define your feedforward pass
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, airfoil_dim, hidden_dims=[64, 128]):
        super(Generator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, airfoil_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(latent_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())

        if len(hidden_dims) != 0:
            layers.append(nn.Linear(hidden_dims[-1], airfoil_dim))
        else:
            layers.append(nn.Linear(latent_dim, airfoil_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # define your feedforward pass
        return self.net(x)

