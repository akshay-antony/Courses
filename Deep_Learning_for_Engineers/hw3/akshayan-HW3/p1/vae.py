from turtle import st
#from cv2 import log
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim=100, latent_dim=16, hidden_dims=[50, 20]):
        super(Encoder, self).__init__()
        # build your model here
        # your model should output a predicted mean and a predicted std of the encoding
        # both should be of dim (batch_size, latent_dim)
        self.layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dim))

        self.common_layers = nn.Sequential(*self.layers)
        if len(hidden_dims) != 0:
            self.mean_layer = nn.Linear(hidden_dims[-1], latent_dim)
            self.std_layer = nn.Linear(hidden_dims[-1], latent_dim)
        else:
            self.mean_layer = nn.Linear(input_dim, latent_dim)
            self.std_layer = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        # define your feedforward pass
        common_out = self.common_layers(x)
        mean = self.mean_layer(common_out)
        std = self.std_layer(common_out)
        return mean, std


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, output_dim=100, hidden_dims=[20, 50]):
        super(Decoder, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, output_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(latent_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            
        if len(hidden_dims) != 0:
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        else:
            layers.append(nn.Linear(latent_dim, output_dim))
        #layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # define your feedforward pass
        out = self.net(x)
        out = torch.tanh(out)
        return out


class VAE(nn.Module):
    def __init__(self, airfoil_dim, latent_dim, hidden_dims_encoder=[50, 20], \
                 hidden_dims_decoder=[20, 50]):
        super(VAE, self).__init__()
        self.enc = Encoder(airfoil_dim, latent_dim, hidden_dims_encoder)
        self.dec = Decoder(latent_dim, airfoil_dim, hidden_dims_decoder)

    def forward(self, x):
        # define your feedforward pass
        mean, logvar = self.enc(x)
        out1 = mean
        out1 = self.reparameterize(mean, logvar)
        return self.dec(out1), mean, logvar

    def decode(self, z):
        # given random noise z, generate airfoils
        return self.dec(z)

    def reparameterize(self, mean, logvar):
        rand_variable = (torch.randn_like(logvar, requires_grad=True)).cuda()
        std = torch.exp(0.5*logvar)
        # print(std.shape, rand_variable.shape, mean.shape)
        out = mean + std * rand_variable
        return out