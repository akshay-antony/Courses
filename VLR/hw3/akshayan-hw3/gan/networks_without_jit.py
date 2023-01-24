from inspect import stack
#from msilib.schema import Shortcut
import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(nn.Module):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.upscale_factor = upscale_factor
        #self.upsample = nn.Upsample(scale_factor=(upscale_factor**2, 1, 1), mode="nearest")
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, padding=padding)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

    #@jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Stack x channel wise upscale_factor^2 times
        # 2. Then re-arrange to form a batch x channel x height*upscale_factor x width*upscale_factor
        # 3. Apply convolution.
        # Hint for 2. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle
        stacked_x = x.repeat((1, int(self.upscale_factor**2), 1, 1))
        shuffled_x = self.pixel_shuffle(stacked_x)
        output = self.conv(shuffled_x)
        return output

class DownSampleConv2D(nn.Module):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(self, input_channels, kernel_size=3, n_filters=128, 
                downscale_ratio=2, padding=0):
        super(DownSampleConv2D, self).__init__()
        self.input_channels = input_channels
        self.down_sample = nn.PixelUnshuffle(downscale_factor=downscale_ratio)
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, padding=padding)
    
    #@jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling
        # 1. Re-arrange to form a batch x channel * upscale_factor^2 x height x width
        # 2. Then split channel wise into batch x channel x height x width Images
        # 3. average the images into one and apply convolution
        # Hint for 1. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle
        rearranged_x = self.down_sample(x)
        split_x = torch.split(rearranged_x, self.input_channels, dim=1)
        combined_x = torch.stack(split_x)
        combined_x = torch.mean(combined_x, dim=0)
        out = self.conv(combined_x)
        return out

class ResBlockUp(nn.Module):
    # TODO 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        self.layers = nn.Sequential(
                        nn.BatchNorm2d(input_channels, eps=1e-05),
                        nn.ReLU(),
                        nn.Conv2d(input_channels, n_filters, kernel_size, padding=(1, 1), bias=False),
                        nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True),
                        nn.ReLU())

        self.residual = UpSampleConv2D(n_filters, kernel_size=(3,3), 
                            n_filters=n_filters, padding=(1,1))

        self.shortcut = UpSampleConv2D(input_channels=input_channels, kernel_size=(1,1),
                            n_filters=n_filters)

    #@jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        layers_out = self.layers(x)
        res_out = self.residual(layers_out)
        shortcut_out = self.shortcut(x)
        out = res_out + shortcut_out
        return out
 

class ResBlockDown(nn.Module):
    # TODO 1.1: Impement Residual Block Downsampler.

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        self.layers = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(input_channels, n_filters, kernel_size, padding=(1, 1), bias=False),
                        nn.ReLU())

        self.residual = DownSampleConv2D(n_filters, kernel_size=(3,3), 
                            n_filters=n_filters, padding=(1,1))

        self.shortcut = DownSampleConv2D(input_channels=input_channels, kernel_size=(1,1),
                            n_filters=n_filters)

    #@jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        layers_out = self.layers(x)
        res_out = self.residual(layers_out)
        shortcut_out = self.shortcut(x)
        out = res_out + shortcut_out
        return out


class ResBlock(nn.Module):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
                        nn.ReLU(), 
                        nn.Conv2d(input_channels, n_filters, kernel_size, padding=(1, 1)),
                        nn.ReLU(), 
                        nn.Conv2d(n_filters, n_filters, kernel_size, padding=(1, 1)))

    #@jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        out = self.layers(x)
        output = out + x
        return output


class Generator(nn.Module):
    # TODO 1.1: Impement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=4096, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """
    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        self.starting_image_size = starting_image_size
        self.dense = nn.Linear(128, 2048, bias=True)
        self.layers = nn.Sequential(
                        ResBlockUp(input_channels=128),
                        ResBlockUp(input_channels=128),
                        ResBlockUp(input_channels=128))
        self.batchnorm = nn.BatchNorm2d(128, eps=1e-05)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(128, 3, kernel_size=3, padding=(1, 1))
        self.tanh = nn.Tanh()

    #@jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        z = self.dense(z.cuda())
        start_image = torch.reshape(z, (z.shape[0], -1, 4, 4))
        out1 = self.layers(start_image)
        out2 = self.tanh(self.conv(self.relu(self.batchnorm(out1))))
        return out2

    #@jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        # Make sure to cast the latents to type half (for compatibility with torch.cuda.amp.autocast)
        latent = torch.randn((n_samples, 128))
        latent = latent.half()
        return self.forward_given_samples(latent)


class Discriminator(nn.Module):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (1): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (2): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (3): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
                            ResBlockDown(input_channels=3), 
                            ResBlockDown(input_channels=128),  
                            ResBlock(input_channels=128),  
                            ResBlock(input_channels=128),  
                            nn.ReLU())

        self.dense = nn.Linear(128, 1)

    #@jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to flatten the output of the convolutional layers before passing to the output layer!
        out1 = self.layers(x)
        out1 = out1.reshape(out1.shape[0], out1.shape[1], -1)
        out1 = torch.sum(out1, dim=2)
        out2 = self.dense(out1)
        return out2