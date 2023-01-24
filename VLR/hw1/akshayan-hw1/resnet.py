from torchvision import models
import torch
import torch.nn as nn 

class ResNet(nn.Module):
	def __init__(self):
		super(ResNet, self).__init__()
		self.resnet = models.resnet18(pretrained=False)
		self.resnet.fc = nn.Linear(512, 20)

	def forward(self, x):
		return self.resnet(x)

