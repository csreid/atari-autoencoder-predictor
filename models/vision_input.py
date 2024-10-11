import torch
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import functional as F
from torch.nn import init as I
from torchvision.transforms import Resize
from torch.nn import (
	Module,
	LSTM,
	Linear,
	Conv2d,
	functional as F,
	Sequential,
	ReLU,
	Flatten,
	BatchNorm2d,
	BatchNorm1d,
	MaxPool2d
)
from torch.utils.data import DataLoader

class VisionInputBlock(Module):
	def __init__(self, channels_in, channels_out, kernel_size, stride=1):
		super().__init__()

		self._conv = Conv2d(
			channels_in,
			channels_out,
			kernel_size=kernel_size,
			stride=stride,
			padding='valid'
		)
		self._norm = BatchNorm2d(channels_out)

	def forward(self, X):
		out = self._conv(X)
		out = F.leaky_relu(out, negative_slope=0.1)
		out = self._norm(out)

		return out

class VisionInput(Module):
	def __init__(self, img_features, pretrained=False, frozen=False):
		super().__init__()

		if pretrained:
			print(f'Ignoring pretraining for simple vision input')

		self._bn = BatchNorm2d(1)
		self._bn_1d = BatchNorm1d(img_features)

		self._cnn_list = [
			VisionInputBlock(1, 4, kernel_size=4, stride=2),
			VisionInputBlock(4, 8, kernel_size=4, stride=2),
			VisionInputBlock(8, 16, kernel_size=4),
			VisionInputBlock(16, 32, kernel_size=4),
			VisionInputBlock(32, 64, kernel_size=4),
			VisionInputBlock(64, 128, kernel_size=4),
			VisionInputBlock(128, 128, kernel_size=4),
			VisionInputBlock(128, 128, kernel_size=4),
			VisionInputBlock(128, 256, kernel_size=4),
			#VisionInputBlock(128, 256, kernel_size=3),
		]

		self._cnn = Sequential(*self._cnn_list)
		self._fc = Linear(73984, img_features)
		I.constant_(self._fc.weight, 0.)
		self._flatten = Flatten()

	def forward(self, X):
		if len(X.shape) == 5:
			# We need to flatten things first
			b, s, c, x, y = X.shape
			res = torch.flatten(X, start_dim=0, end_dim=1)
		else:
			res = X

		res = self._bn(res)
		for c in self._cnn_list:
			res = c(res)
		res = self._flatten(res)
		res = self._fc(res)
		res = self._bn_1d(res)
		res = torch.tanh(res)

		if len(X.shape) == 5:
			res = torch.unflatten(res, dim=0, sizes=(b, s))

		return res
