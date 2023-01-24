# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# Modified by Sudeep Dasari
# --------------------------------------------------------
from __future__ import print_function
from fileinput import filename

import torch
import numpy as np
import os
import utils
from voc_dataset import VOCDataset
from torch.utils.tensorboard import SummaryWriter


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    # TODO: Q2.2 Implement code for model saving
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    torch.save(model.state_dict(), filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    # TODO Q1.5: Initialize your tensorboard writer here!
    filename_tensorboard = "runs/" + model_name
    writer = SummaryWriter(filename_tensorboard) 
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)
    

    cnt = 0
    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            # Get a batch of data
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate the loss
            # TODO Q1.4: your loss for multi-label classification
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = wgt)
            loss = loss_fn(output, target)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                # TODO Q1.5: Log training loss to tensorboard
                writer.add_scalar("Training loss", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                # TODO Q3.2: Log histogram of gradients
                for name, params in model.named_parameters():
                    writer.add_histogram(name, params.grad.data.cpu().numpy(), cnt)

            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                # TODO Q1.5: Log MAP to tensorboard
                writer.add_scalar("Testing MAP", map, cnt)
                model.train()
            cnt += 1
            writer.flush()
            torch.cuda.empty_cache()
            
        # TODO Q3.2: Log Learning rate
        if scheduler is not None:
            scheduler.step()
            curr_lr = scheduler.get_last_lr()
            writer.add_scalar("Learning Rate", curr_lr[0], epoch)
            #writer.add_scalar("Learning Rate", args.lr, epoch)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    writer.close()
    return ap, map
