from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from operator import mod

import os
from pickletools import optimize
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL, calculate_ap, get_box_data
from PIL import Image, ImageDraw


# hyper-parameters
# ------------
start_step = 0
end_step = 20000
lr_decay_steps = {150000}
lr_decay = 1. / 10
rand_seed = 1024

lr = 0.0001
momentum = 0.9
weight_decay = 0.0005
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load datasets and create dataloaders
image_size = 512
train_dataset = VOCDataset(image_size=image_size, top_n=500)
val_dataset = VOCDataset(image_size=image_size, top_n=500)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,   # batchsize is one for this implementation
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    sampler=None,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True)


# Create network and initialize
net = WSDDN(classes=train_dataset.CLASS_NAMES, roi_size=(6, 6))
print(net)

if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()

for name, param in pret_net.items():
    print(name)
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue


# Move model to GPU and set train mode
net.load_state_dict(own_state)
net.cuda()
net.train()

# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)
for i, (name,param) in enumerate(net.features.named_parameters()):
    param.requires_grad = False
    print(name, param.requires_grad)

optimizer = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
#optimizer = torch.optim.Adam(net.parameters(), lr)
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=lr_decay)
# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
disp_interval = 10
val_interval = 1000

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

def plot_boxes(boxes, labels, image, class_names, image_size=512, image_name = None):
    boxes = boxes.detach().cpu().numpy()
    boxes /= image_size
    labels = labels.detach().cpu().numpy()
    labels = np.int16(labels)
    boxes = boxes.tolist()
    labels = labels.tolist()
    image = tensor_to_PIL(image[0])
    class_id_to_label = dict(enumerate(class_names))

    img = wandb.Image(image, boxes={
        "predictions": {
            "box_data": get_box_data(labels, boxes),
            "class_labels": class_id_to_label,
        },
    })
    wandb.log({image_name: img})

def test_net(model, val_loader=None, thresh=0.05, iou_thresh=0.5, image_size=512, epoch=None, plot_intervals=None):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """
    all_gt_boxes = torch.zeros((0, 6))
    all_pred_boxes = torch.zeros((0, 7))
    dict_class_tp = {}
    dict_class_fp = {}
    dict_class_scores = {}
    dict_class_total_gt_labels = {}
    for i in range(20):
        dict_class_tp[i] = torch.zeros((0))
        dict_class_fp[i] = torch.zeros((0))
        dict_class_scores[i] = torch.zeros((0))
        dict_class_total_gt_labels[i] = 0

    for iter, data in enumerate(val_loader):
        # one batch = data for one image
        image           = data['image'].cuda()
        target          = data['label'].cuda()
        wgt             = data['wgt'].cuda()
        rois            = data['rois'].squeeze().cuda()
        gt_boxes        = data['gt_boxes'].reshape(-1, 4)
        gt_class_list   = data['gt_classes'].reshape(-1)
        #TODO: perform forward pass, compute cls_probs
        cls_probs = model(image, rois, target)

        for i in range(gt_boxes.shape[0]):
            modified_boxes = torch.cat([torch.tensor([iter, gt_class_list[i]]), gt_boxes[i]]).reshape(1, -1)
            all_gt_boxes = torch.cat([all_gt_boxes, modified_boxes], dim=0)

        # TODO: Iterate over each class (follow comments)
        for class_num in range(20):            
            # get valid rois and cls_scores based on thresh
            # use NMS to get boxes and scores
            boxes, scores = nms(rois, cls_probs[:, class_num])
            
            for i in range(boxes.shape[0]):
                modified_pred_boxes = torch.cat([torch.tensor([iter, class_num, scores[i]]), boxes[i]]).reshape(1, -1)
                all_pred_boxes = torch.cat([all_pred_boxes, modified_pred_boxes], dim=0)


        #TODO: visualize bounding box predictions when required
            # if iter%1000 == 0 and boxes.shape[0] > 0:
            #     plot_boxes(boxes, scores, image, net.classes, torch.full((boxes.shape[0],), class_num), image_size=image_size)
    
    #TODO: Calculate mAP on test set
    AP = calculate_ap(all_pred_boxes, all_gt_boxes)
    mAP = 0 if len(AP) == 0 else sum(AP) / len(AP)
    #mAP = calculate_ap(all_pred_boxes, all_gt_boxes)
    return mAP.item(), AP

def plot_random_images(model, class_names, train_dataset, number_of_images=10, epoch=0):
    np.random.seed(40)
    rand_idxs = np.random.randint(0, len(train_dataset), (number_of_images))
    for rand_ind in rand_idxs:
        data = train_dataset[rand_ind]
        image           = data['image'].cuda()
        target          = data['label'].cuda()
        wgt             = data['wgt'].cuda()
        rois            = data['rois'].squeeze().cuda()
        gt_boxes        = data['gt_boxes'].cuda()
        gt_class_list   = data['gt_classes'].cuda()

        image = image.unsqueeze(0)
        target = target.unsqueeze(0)
        cls_probs = model(image, rois, target)
        total_boxes = torch.zeros((0,4))
        total_labels = torch.zeros((0))
        for class_num in range(20):    
            boxes, scores = nms(rois, cls_probs[:, class_num])
            total_boxes = torch.cat([total_boxes, boxes.reshape(-1, 4)], dim=0)
            total_labels = torch.cat([total_labels, torch.full((boxes.shape[0],), class_num)], dim=0)
        plot_boxes(total_boxes, total_labels, image, net.classes, image_size=image_size, image_name = "Random image for epoch " + str(epoch))

if __name__ == '__main__':
    plot_intervals = set()
    wandb.init(project="vlr2_task2", reinit=True)
    epochs = 6 
    plot_intervals.add(0)
    plot_intervals.add(epochs-1)
    # training
    train_loss = 0
    tp, tf, fg, bg = 0., 0., 0, 0
    step_cnt = 0
    re_cnt = False
    disp_interval = 10
    val_interval = len(train_loader) - 1
    train_loss_every = 500
    
    for epoch in range(epochs):
        train_loss = 0
        tp, tf, fg, bg = 0., 0., 0, 0
        step_cnt = 0
        losses = AverageMeter()
        for iter, data in enumerate(train_loader):
            #TODO: get one batch and perform forward pass
            # one batch = data for one image
            image           = data['image'].cuda()
            target          = data['label'].cuda()
            wgt             = data['wgt'].cuda()
            rois            = data['rois'].squeeze().cuda()
            gt_boxes        = data['gt_boxes'].cuda()
            gt_class_list   = data['gt_classes'].cuda()
            
            #TODO: perform forward pass - take care that proposal values should be in pixels for the fwd pass
            # also convert inputs to cuda if training on GPU
            cls_prob = net(image, rois, target)

            # backward pass and update
            loss = net.loss    
            losses.update(loss.item())
            train_loss += loss.item()
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #plot_random_images(net, train_dataset.CLASS_NAMES, train_dataset, 10, epoch=epoch)
            #TODO: evaluate the model every N iterations (N defined in handout)
            if iter%val_interval == 0 and iter != 0:
                net.eval()
                map, AP = test_net(net, val_loader, plot_intervals=plot_intervals, image_size=image_size, epoch=epoch)
                print("AP ", map)
                wandb.log({"mAP": map})
                for class_ap, class_name in zip(AP, train_dataset.CLASS_NAMES):
                    wandb.log({"AP of " + class_name: class_ap})
                net.train()

            if iter%train_loss_every == 0 and iter != 0:
                wandb.log({"train/loss_avg": losses.avg}) #, step=len(train_loader)*epoch+iter)
                print("Epoch: ", epoch, " loss: ", losses.avg)

        if epoch in plot_intervals:
            plot_random_images(net, train_dataset.CLASS_NAMES, train_dataset, 20, epoch=epoch)
        #TODO: Perform all visualizations here
        #The intervals for different things are defined in the handout
        scheduler.step()




