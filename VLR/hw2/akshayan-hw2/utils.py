import os
import random
import time
import copy
import wandb

from PIL import  Image
import matplotlib
from collections import Counter
from sklearn.metrics import average_precision_score
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt




def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    valid_threhold = torch.where(confidence_score > threshold)[0]
    bounding_boxes = bounding_boxes[valid_threhold, :]
    confidence_score = confidence_score[valid_threhold]
    boxes = []
    scores = []
    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    confidence_score = confidence_score.detach().cpu().numpy()

    while bounding_boxes.shape[0] != 0:
        confidence_max_idx = np.argmax(confidence_score)
        boxes.append(bounding_boxes[confidence_max_idx, :])
        scores.append(confidence_score[confidence_max_idx])
        reference_box = bounding_boxes[confidence_max_idx, :]

        valid_indx = []
        for i in range(bounding_boxes.shape[0]):
            if iou(reference_box, bounding_boxes[i]) < 0.3:
                valid_indx.append(i)
        bounding_boxes = bounding_boxes[valid_indx, :]
        confidence_score = confidence_score[valid_indx]

    boxes = torch.FloatTensor(boxes)
    scores = torch.FloatTensor(scores)
    return boxes, scores


#TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    if box1[0] > box2[2] or box1[2] < box2[0] or box1[1] > box2[3] or box1[3] < box2[1]:
        return 0 
    x1_common = max(box1[0], box2[0])
    x2_common = min(box1[2], box2[2])
    y1_common = max(box1[1], box2[1])
    y2_common = min(box1[3], box2[3])

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    common_area = (x2_common - x1_common) * (y2_common - y1_common)
    iou = common_area / (area1 + area2 - common_area)
    return iou

def nms_vector(bounding_boxes, confidence_score, threshold=0.05, nms_threshold=0.3):
    valid_threhold = torch.where(confidence_score > threshold)[0]
    bounding_boxes = bounding_boxes[valid_threhold, :]
    confidence_score = confidence_score[valid_threhold]
    boxes = []
    scores = []
    bounding_boxes = bounding_boxes.detach().cpu()
    confidence_score = confidence_score.detach().cpu()

    while bounding_boxes.shape[0] != 0:
        confidence_max_idx = torch.argmax(confidence_score)
        boxes.append(bounding_boxes[confidence_max_idx, :])
        scores.append(confidence_score[confidence_max_idx])
        reference_box = bounding_boxes[confidence_max_idx, :]
        print(reference_box.shape, bounding_boxes.shape)
        iou_tensor = iou_vector(reference_box, bounding_boxes)
        valid_idx = torch.where(iou_tensor < torch.tensor([nms_threshold]))[0]
        print(valid_idx)
        bounding_boxes = bounding_boxes[valid_idx[0], :]
        confidence_score = confidence_score[valid_idx[0]]
    
    
    boxes = torch.FloatTensor(boxes)
    scores = torch.FloatTensor(scores)
    return boxes, scores

def iou_vector(reference_box, box_vector):
    x1_common = torch.max(box_vector[:, 0], reference_box[0])
    x2_common = torch.min(box_vector[:, 2], reference_box[2])
    y1_common = torch.max(box_vector[:, 1], reference_box[1])
    y2_common = torch.min(box_vector[:, 3], reference_box[3])

    common_area = torch.max(torch.tensor([0]), x2_common - x1_common) \
                  * torch.max(torch.tensor([0]), y2_common - y1_common)

    area1 = (box_vector[:,2] - box_vector[:,0]) * (box_vector[:, 3] - box_vector[:, 1])
    area2 = (reference_box[2] - reference_box[0]) * (reference_box[3] - reference_box[1])
    iou = common_area / (area1 + area2 - common_area)
    print(iou.shape)
    return iou

def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id" : classes[i],
        } for i in range(len(classes))
        ]

    return box_list



def get_box_data_one_class(class_num, bbox_coordinates, image_size=512):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0] / image_size,
                "minY": bbox_coordinates[i][1] / image_size,
                "maxX": bbox_coordinates[i][2] / image_size,
                "maxY": bbox_coordinates[i][3] / image_size,
            },
            "class_id" : class_num,
        } for i in range(bbox_coordinates.shape[0])
        ]

    return box_list


def calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.5):
    AP = []
    for class_num in range(20):
        valid_gt_boxes = torch.zeros((0, 6))
        valid_pred_boxes = torch.zeros((0, 7))

        valid_gt_boxes_ind = torch.where(gt_boxes[:,1] == class_num)
        valid_gt_boxes = gt_boxes[valid_gt_boxes_ind]
        
        valid_pred_boxes_ind = torch.where(pred_boxes[:, 1] == class_num)
        valid_pred_boxes = pred_boxes[valid_pred_boxes_ind]
        
        pred_ind = torch.argsort(valid_pred_boxes[:,2], descending=True)
        valid_pred_boxes = valid_pred_boxes[pred_ind]

        FP = torch.zeros((valid_pred_boxes.shape[0]))
        TP = torch.zeros((valid_pred_boxes.shape[0]))
        total_gts = valid_gt_boxes.shape[0]
        if total_gts == 0:
            AP.append(torch.tensor([0]))
            continue

        for i in range(valid_pred_boxes.shape[0]):
            curr_valid_gt_boxes_ind = torch.where(valid_gt_boxes[:,0] == valid_pred_boxes[i,0])
            curr_valid_gt_boxes = valid_gt_boxes[curr_valid_gt_boxes_ind] 

            taken_gt_boxes = set()

            best_iou = 0
            for j in range(curr_valid_gt_boxes.shape[0]):
                curr_iou = iou(curr_valid_gt_boxes[j, 2:].reshape(-1), valid_pred_boxes[i, 3:].reshape(-1))
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_gt_idx = j 
            
            if best_iou >= iou_threshold:
                if best_gt_idx in taken_gt_boxes:
                    FP[i] = 1 
                else:
                    taken_gt_boxes.add(best_gt_idx)
                    TP[i] = 1 
            else:
                FP[i] = 1 

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / total_gts
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        ap = 0
        for i in range(precisions.shape[0]):
            precisions[i] = torch.tensor([max(precisions[i].item(), torch.max(precisions[i:]).item())])
            if i >= 1:
                ap += precisions[i] * (recalls[i] - recalls[i-1]) 


        AP.append(ap)
        #AP.append(torch.trapz(precisions, recalls))
    return AP


def plot_2_epochs(heatmap, image, target, class_names, epoch):
    image = image.detach().cpu()
    heatmap = heatmap.detach().cpu()
    target = target.detach().cpu()
    
    np.random.seed(32)
    rand_ind = np.random.randint(0, target.shape[0], (1))

    image = image[rand_ind].squeeze(0)
    heatmap = heatmap[rand_ind].squeeze(0)
    target = target[rand_ind].squeeze(0)
    image = tensor_to_PIL(image)
    heatmaps = []
    image_name = []
    for i in range(target.shape[0]):
        if target[i] == 1:
            curr_heatmap = heatmap[i].squeeze().numpy()
            file_name = "temp_epoch_2" + ".png"
            plt.imsave(file_name, curr_heatmap, cmap='jet')
            curr_heatmap = Image.open(file_name)

            curr_heatmap = curr_heatmap.resize(image.size)
            image_name.append(class_names[i])
            heatmaps.append(curr_heatmap)
    wandb.log({"Train input image": wandb.Image(image, caption="Input image")})
    wandb.log({"Train heatmaps per epoch for epoch: " + str(epoch):\
             [wandb.Image(image, caption=caption) \
             for image, caption in zip(heatmaps, image_name)]})

def plot_image(img, heatmap, target, class_names, epoch, isval=False, heatmap_name="Train heatmaps"):
    img = img.detach().cpu()
    heatmap = heatmap.detach().cpu() #.numpy()
    target = target.detach().cpu().numpy()
    img = tensor_to_PIL(img)
    img.show()
    wandb_img = wandb.Image(img, caption="Original input image")
    wandb.log({'Train Input image': wandb_img})
    if isval:
        wandb.log({'Validation Input image': wandb_img})
    else:
        wandb.log({'Input image': wandb_img})

    heatmaps = []
    image_name = []
    for i in range(target.shape[0]):
        if target[i] == 1:
            curr_heatmap = heatmap[i].squeeze().numpy()
            file_name = "temp" + str(i) + ".png"
            plt.imsave(file_name, curr_heatmap, cmap='jet')
            curr_heatmap = Image.open(file_name)
            #curr_heatmap = scale_heatmap(heatmap[i].squeeze())
            curr_heatmap = curr_heatmap.resize((512, 512))
            image_name.append(class_names[i])
            heatmaps.append(curr_heatmap)
    if isval:
        wandb.log({"Validation heatmaps": [wandb.Image(image, caption=caption) for image, caption in zip(heatmaps, image_name)]})
    else:
        wandb.log({heatmap_name: [wandb.Image(image, caption=caption) for image, caption in zip(heatmaps, image_name)]})

def scale_heatmap(img):
    #img = torch.sigmoid(img)
    # img -= img.min()
    # img /= img.max()
    # img *= 255
    #img = torch.sigmoid(img)
    img = transforms.ToPILImage()(img).convert("RGB")
    # cm_hot = matplotlib.cm.get_cmap('jet')
    # colored_image = cm_hot(img)
    # colored_image = np.asarray(colored_image)
    # img = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
    return img
