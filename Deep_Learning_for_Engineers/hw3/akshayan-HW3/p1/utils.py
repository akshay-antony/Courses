'''
Utility functions 
Free from to add functions if needed
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def plot_airfoils(airfoil_x, airfoil_y, fname):
    '''
    plot airfoils: no need to modify 
    '''
    idx = 0
    fig, ax = plt.subplots(nrows=4, ncols=4)
    for row in ax:
        for col in row:
            col.scatter(airfoil_x, airfoil_y[idx, :], s=0.6, c='black')
            col.axis('off')
            col.axis('equal')
            idx += 1
    #plt.show()
    plt.savefig(fname)
    plt.show()

def gan_loss_disc(real_pred, fake_pred):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    real_labels = torch.ones_like(real_pred)
    fake_labels = torch.zeros_like(fake_pred)
    total_pred = torch.cat([real_pred.reshape(-1), fake_pred.reshape(-1)])
    total_labels = torch.cat([real_labels.reshape(-1), fake_labels.reshape(-1)])
    loss = loss_fn(total_pred, total_labels)
    return loss 

def gan_loss_gen(fake_pred):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    fake_labels = torch.ones_like(fake_pred)
    loss = loss_fn(fake_pred, fake_labels)
    return loss