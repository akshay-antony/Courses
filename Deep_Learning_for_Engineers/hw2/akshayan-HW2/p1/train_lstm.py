import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from dataset import FlowDataset
from lstm import FlowLSTM
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def validate(model, test_loader, writer, epoch):
    l1_err, l2_err = 0, 0
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    model.eval()


    for n_batch, (in_batch, label) in enumerate(test_loader):
        in_batch, label = in_batch.to(device), label.to(device)
        pred = model.test(in_batch)

        l1_err += l1_loss(pred, label).item()
        l2_err += l2_loss(pred, label).item()

    writer.add_scalar("Validation total L1 error", l1_err, epoch)
    writer.add_scalar("Validation total L2 error", l2_err, epoch)


def main():
    # check if cuda available
    parser = argparse.ArgumentParser(description="Flow prediction using LSTM")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--validate_epoch', type=int, default=5)
    args = parser.parse_args()

    args.hidden_size = [128, 256, 512]
    # define dataset and dataloader
    train_dataset = FlowDataset(mode='train')
    test_dataset = FlowDataset(mode='test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # hyper-parameters
    input_size = 17 # do not change input size

    model = FlowLSTM(
        input_size=input_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers, 
        dropout=args.dropout
    ).to(device)

    # define your LSTM loss function here
    l2_loss_func = nn.MSELoss()
    l1_loss_func = nn.L1Loss()
    # define optimizer for lstm model
    optim = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=25, gamma=0.1)
    writer = SummaryWriter()

    for epoch in tqdm(range(args.epochs)):
        epoch_loss = 0
        num_batches = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for n_batch, (in_batch, label) in loop:
            in_batch, label = in_batch.to(device), label.to(device)
           # train LSTM
            model.train()
            out = model(in_batch)
            # calculate LSTM loss
            l1_loss = l1_loss_func(out, label)
            l2_loss = l2_loss_func(out, label)
            loss = l2_loss #l1_loss * 1e-3 + l2_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            num_batches += label.shape[0]

            # print loss while training
            if (n_batch) % 10 == 9:
                loop.set_description(f"Epoch [{epoch}/{args.epochs}]")
                loop.set_postfix(loss = loss.item())
            
            torch.cuda.empty_cache()
        writer.add_scalar("MSE train loss", (epoch_loss / num_batches), epoch)
        if epoch % args.validate_epoch == args.validate_epoch-1:
            validate(model, test_loader, writer, epoch)
        scheduler.step()

    torch.save(model.state_dict(), "model.ckpt")
    writer.close()
    # test trained LSTM model
    l1_err, l2_err = 0, 0
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for n_batch, (in_batch, label) in enumerate(test_loader):
            in_batch, label = in_batch.to(device), label.to(device)
            pred = model.test(in_batch)

            l1_err += l1_loss(pred, label).item()
            l2_err += l2_loss(pred, label).item()

    print("Test L1 error:", l1_err)
    print("Test L2 error:", l2_err)

    # visualize the prediction comparing to the ground truth
    if device == 'cpu':
        pred = pred.detach().numpy()[0,:,:]
        label = label.detach().numpy()[0,:,:]
    else:
        pred = pred.detach().cpu().numpy()[0,:,:]
        label = label.detach().cpu().numpy()[0,:,:]

    r = []
    num_points = 17
    interval = 1./num_points
    x = int(num_points/2)
    for j in range(-x,x+1):
        r.append(interval*j)

    plt.figure()
    for i in range(1, len(pred)):
        c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
        plt.plot(pred[i], r, label='t = %s' %(i), c=c)
    plt.xlabel('velocity [m/s]')
    plt.ylabel('r [m]')
    plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    #plt.show()
    plt.savefig("pred.png")

    plt.figure()
    for i in range(1, len(label)):
        c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
        plt.plot(label[i], r, label='t = %s' %(i), c=c)
    plt.xlabel('velocity [m/s]')
    plt.ylabel('r [m]')
    plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    plt.savefig("gt.png")
    plt.show()


if __name__ == "__main__":
    main()

