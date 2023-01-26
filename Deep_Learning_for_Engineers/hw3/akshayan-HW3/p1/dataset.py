import os
import torch
import numpy as np
from torch.utils.data import Dataset


class AirfoilDataset(Dataset):
	'''
	airfoil dataset: no need to modify
	'''
	def __init__(self, path='./airfoils'):
		super(AirfoilDataset, self).__init__()
		self._X = []	# x coordinates of all airfoils (shared)
		self._Y = []	# y coordinates of all airfoils
		self.names = []	# name of all airfoils
		self.norm_coeff = 0	# normalization coeff to scale y to [-1, 1]
		airfoil_fn = [afn for afn in os.listdir(path) if afn.endswith('.dat')]

		# get x coordinates of all airfoils
		with open(os.path.join(path, airfoil_fn[0]), 'r', encoding="utf8", errors='ignore') as f:
			raw_data = f.readlines()
			for idx in range(len(raw_data)):
				raw_xy = raw_data[idx].split(' ')
				while "" in raw_xy:
					raw_xy.remove("")
				self._X.append(float(raw_xy[0]))
		self._X = np.array(self._X)

		# get y coordinates of each airfoils
		for idx, fn in enumerate(airfoil_fn):
			with open(os.path.join(path, fn), 'r', encoding="utf8", errors='ignore') as f:
				self.names.append(fn[:-10])
				raw_data = f.readlines()
				airfoil = np.empty(self._X.shape[0])
				for i in range(len(raw_data)):
					raw_xy = raw_data[i].split(' ')
					while "" in raw_xy:
						raw_xy.remove("")
					curr_y = float(raw_xy[1])
					airfoil[i] = curr_y
					self.norm_coeff = max(self.norm_coeff, np.abs(curr_y))
				self._Y.append(airfoil)

		self._Y = np.array([airfoil / self.norm_coeff for airfoil in self._Y], dtype=np.float32)

	def get_x(self):
		'''
		get shared x coordinates
		'''
		return self._X

	def get_y(self):
		'''
		get y coordinates of all airfoils
		'''
		return self._Y

	def __getitem__(self, idx):
		return self._Y[idx], self.names[idx]
		
	def __len__(self):
		return len(self._Y)

