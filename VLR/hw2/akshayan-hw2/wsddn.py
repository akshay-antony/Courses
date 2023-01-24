from multiprocessing import reduction
from types import new_class
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import numpy as np

from torchvision.ops import roi_pool, roi_align


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None, roi_size=(6,6)):
        super(WSDDN, self).__init__()
        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        self.roi_size = roi_size
        #TODO: Define the WSDDN model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        
        self.roi_pool   = torchvision.ops.roi_pool
        self.classifier = nn.Sequential(
                            nn.Linear(256*self.roi_size[0]*self.roi_size[1], 4096), 
                            nn.ReLU(inplace=True), 
                            nn.Linear(4096, 4096), 
                            nn.ReLU(inplace=True))

        self.score_fc   = nn.Sequential(
                            nn.Linear(4096, self.n_classes))
                            #nn.Softmax(dim=1))

        self.bbox_fc    = nn.Sequential(
                            nn.Linear(4096, self.n_classes))
                            #nn.Softmax(dim=0))

        # loss
        self.cross_entropy = None


    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):
        

        #TODO: Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        conv_features = self.features(image)
        h, w = conv_features.shape[2], conv_features.shape[3]
        spp_output = self.roi_pool(conv_features, [rois], self.roi_size, h/image.shape[2])
        spp_output = spp_output.reshape(spp_output.shape[0], -1)
        classifier_ouput = self.classifier(spp_output)
        class_scores = self.score_fc(classifier_ouput)
        class_scores = nn.functional.softmax(class_scores, dim=1)
        
        bbox_scores = self.bbox_fc(classifier_ouput)
        bbox_scores = nn.functional.softmax(bbox_scores, dim=0)

        cls_prob = class_scores * bbox_scores

        if self.training:
            label_vec = gt_vec.view(1, -1)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        
        return cls_prob

    
    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        #TODO: Compute the appropriate loss using the cls_prob that is the
        #output of forward()
        #Checkout forward() to see how it is called

        loss_fn = nn.BCELoss(reduction='sum')
        pred = torch.sum(cls_prob, dim=0, keepdims=True)
        pred = torch.clamp(pred, 0, 1)
        loss = nn.functional.binary_cross_entropy(pred, label_vec, reduction='sum')
        return loss
