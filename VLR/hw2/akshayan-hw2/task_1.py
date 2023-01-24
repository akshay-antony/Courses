import argparse
from cgi import test
from operator import iadd
import os
from pickletools import optimize
import shutil
import time
import sys
from matplotlib.pyplot import sca
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sklearn
import sklearn.metrics
import math
from sympy import im

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


from AlexNet import *
from voc_dataset import *
from utils import *

import wandb
USE_WANDB = True # use flags, wandb is not convenient for debugging


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', \
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N', \
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', \
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', \
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', \
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', \
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', \
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',  \
                    help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=2, type=int, metavar='N',  \
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',  \
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', \
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',  \
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int, \
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, \
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,  \
                    help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    args.pretrained = True

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    # you can also use PlateauLR scheduler, which usually works well
    criterion = nn.BCELoss()
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                                weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Data loading code
    
    #TODO: Create Datasets and Dataloaders using VOCDataset - Ensure that the sizes are as required
    # Also ensure that data directories are correct - the ones use for testing by TAs might be different
    # Resize the images to 512x512

    train_dataset = VOCDataset(split='trainval', image_size=512, top_n=10)
    val_dataset = VOCDataset(split='test', image_size=512, top_n=10)
    print(args.batch_size)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    
    
    # TODO: Create loggers for wandb - ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB:
        wandb.init(project="vlr2_task1", reinit=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, train_dataset.CLASS_NAMES)
        scheduler_step.step()

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, train_dataset.CLASS_NAMES, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)




#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, class_names, plot_interval=1, plot_epochs=[1, 15, 30]):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()
    number_of_batches = len(train_loader)
    plot_interval_list = set()
    plot_train_heatmap_list = set()
    plot_train_heatmap_list.add(number_of_batches//2)
    plot_train_heatmap_list.add(number_of_batches-1)

    for i in range(plot_interval):
        plot_interval_list.add((i+1)*(number_of_batches // (plot_interval + 1)))
    
    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO: Get inputs from the data dict
        img = data['image'].cuda()
        target = data['label'].cuda()

        # TODO: Get output from model
        heatmap, imoutput = model(img)
        # imoutput = F.max_pool2d(heatmap, (heatmap.shape[2], heatmap.shape[3]), stride=(1,1), padding=(0,0))
        # imoutput = imoutput.reshape((imoutput.shape[0], -1))

        # TODO: Perform any necessary functions on the output such as clamping
        imoutput = torch.sigmoid(imoutput)
        # TODO: Compute loss using ``criterion``
        loss = criterion(imoutput, target)


        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), img.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)


        # TODO:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == args.print_freq-1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize/log things as mentioned in handout
        wandb.log({'epoch': epoch, 'train/avg_losses': losses.avg, 'train/loss': loss.item(), 
                'train/curr_metric1': avg_m1.val, 'train/curr_metric2': avg_m2.val, 
                'train/avg_metric1': avg_m1.avg, 'train/avg_metric2': avg_m2.avg}) #, step=number_of_batches*epoch+i+1)
        
        #TODO: Visualize at appropriate intervals
        if i in plot_train_heatmap_list:
            plot_2_epochs(heatmap, img, target, class_names, epoch)

    if (epoch+1) in plot_epochs:
        np.random.seed(65)
        rand_idxs = np.random.randint(0, img.shape[0], (10))
        #plot_2_epochs(heatmap, img, target, class_names, epoch)
        for rand_idx in rand_idxs:
            plot_image(img[rand_idx].squeeze(0), heatmap[rand_idx].squeeze(0), target[rand_idx].squeeze(0), class_names, 
                       epoch=epoch, heatmap_name="Image heatmap for epoch " + str(epoch))

    # if epoch == args.epochs-1:
    #     np.random.seed(20)
    #     rand_idxs = np.random.randint(0, img.shape[0], (3))

    #     for rand_idx in rand_idxs:
    #         plot_image(img[rand_idx].squeeze(0), heatmap[rand_idx].squeeze(0), target[rand_idx].squeeze(0),
    #                   class_names, epoch=epoch, heatmap_name="Final random 3 heatmaps")

def validate(val_loader, model, criterion, class_names, epoch = 0, plot_interval=2):
    number_of_batches = len(val_loader)
    plot_interval_list = set()
    for i in range(plot_interval):
        plot_interval_list.add((i+1)*(number_of_batches // (plot_interval + 1)))

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):

        # TODO: Get inputs from the data dict
        img = data['image'].cuda()
        target = data['label'].cuda()


        # TODO: Get output from model
        heatmap, imoutput = model(img)
        # imoutput = F.max_pool2d(heatmap, (heatmap.shape[2], heatmap.shape[3]), stride=(1,1), padding=(0,0))
        # imoutput = imoutput.reshape((imoutput.shape[0], -1))
        # TODO: Perform any necessary functions on the output such as clamping
        imoutput = torch.sigmoid(imoutput)
        # TODO: Compute loss using ``criterion``
        loss = criterion(imoutput, target)
        
        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), img.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == args.print_freq-1:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

            #TODO: Visualize things as mentioned in handout
            wandb.log({'val/losses': losses.avg, 'val/loss': loss.item(), 'val/avg_metric1': avg_m1.avg, 'val/curr_metric1': avg_m1.val, 'val/avg_metric2': avg_m2.avg, 'val/curr_metric2': avg_m2.val})
    if epoch == args.epochs-1:
        np.random.seed(85)
        rand_idxs = np.random.randint(0, img.shape[0], (10))

        for rand_idx in rand_idxs:
            plot_image(img[rand_idx].squeeze(0), heatmap[rand_idx].squeeze(0), target[rand_idx].squeeze(0),
                      class_names, epoch=epoch, heatmap_name="Final random 3 heatmaps")
        #TODO: Visualize at appropriate intervals
        # if i in plot_interval_list:
        #     np.random.seed(10)
        #     rand_idx = np.random.randint(0, img.shape[0], (1))
        #     plot_image(img[rand_idx].squeeze(0), heatmap[rand_idx].squeeze(0), target[rand_idx].squeeze(0), class_names, True)

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg
    

# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_ap(output, target):
    num_classes = target.shape[1]
    ap = []
    for class_id in range(num_classes):
        output_req = output[:, class_id]
        target_req = target[:, class_id]
        output_req = output_req - 1e-5*target_req
        curr_ap = sklearn.metrics.average_precision_score(target_req, output_req, average=None)
        if not math.isnan(curr_ap):
            ap.append(curr_ap)
    return sum(ap) / len(ap)

def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    target = target.detach().cpu().numpy()
    output = output.detach().cpu().numpy()
    num_classes = target.shape[1]
    ap = []
    for class_id in range(num_classes):
        output_req = output[:, class_id].astype('float32')
        target_req = target[:, class_id].astype('float32')
        output_req = output_req - 1e-5*target_req
        if np.sum(target_req) == 0:
            #ap.append(0)    
            continue
        curr_ap = sklearn.metrics.average_precision_score(target_req, output_req, average=None)
        if not math.isnan(curr_ap):
            ap.append(curr_ap)
    return sum(ap) / (len(ap) if len(ap) > 0 else 1)


def metric2(output, target):
    #TODO: Ignore for now - proceed till instructed
    target = target.detach().cpu().numpy()
    output = output.detach().cpu().numpy()
    output = np.where(output >= 0.5, 1, 0)    
    recall = sklearn.metrics.recall_score(target, output, average='weighted')
    return recall

def adjust_learning_rate(optimizer, epoch):
    if epoch % 30 == 29:
        args.lr /= 10
        for g in optimizer.param_groups:
            g['lr'] = args.lr

if __name__ == '__main__':
    main()
