import torch
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import Module, LSTM, Linear, ConvTranspose2d, functional as F, Sequential, ReLU, Sigmoid, BatchNorm1d, BatchNorm2d, MaxUnpool2d

class VisionReconstructorBlock(Module):
	def __init__(self, channels_in, channels_out, kernel_size, stride=1):
		super().__init__()

		self._ct = ConvTranspose2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride)
		self._norm = BatchNorm2d(channels_in)

	def forward(self, X, output_size=None):
		out = self._norm(X)
		out = self._ct(out, output_size=output_size)
		out = F.leaky_relu(out, negative_slope=0.1)

		return out

class VisionReconstructor(Module):
	def __init__(self, img_features):
		super().__init__()

		self.fc = Linear(img_features, 73984)
		self._deconv_list = [
			VisionReconstructorBlock(256, 128, 4),
			VisionReconstructorBlock(128, 128, 4),
			VisionReconstructorBlock(128, 128, 4),
			VisionReconstructorBlock(128, 64, 4),
			VisionReconstructorBlock(64, 32, 4),
			VisionReconstructorBlock(32, 16, 4),
			VisionReconstructorBlock(16, 8, 4),
			VisionReconstructorBlock(8, 4, 4, stride=2),
			VisionReconstructorBlock(4, 1, 4, stride=2),
		]
		self._deconv = Sequential(*self._deconv_list)
		self._bn = BatchNorm2d(1)

	def forward(self, X):
		output_shapes = [
			20,
			23,
			26,
			29,
			32,
			35,
			38,
			79,
			160,
		]

		out = self.fc(X)
		out = F.leaky_relu(out, negative_slope=0.1)
		out = out.reshape(-1, 256, 17, 17)
		for idx, dec in enumerate(self._deconv_list):
			out = dec(
				out,
				output_size=(output_shapes[idx], output_shapes[idx])
			)

		return out
