import torch
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import Module, LSTM, Linear, ConvTranspose2d, functional as F, Sequential, ReLU, Sigmoid, BatchNorm1d, BatchNorm2d

from models.vision_input import VisionInput
from models.vision_reconstructor import VisionReconstructor

class VisionAutoencoder(Module):
	def __init__(self, features):
		super().__init__()
		self._n_features = features

		self._embedder = VisionInput(features)
		self._recon = VisionReconstructor(features)

	def forward(self, X, return_embedding=False):
		emb = self._embedder(X)
		out = self._recon(emb)

		if not return_embedding:
			return out

		return out, emb

	def embed(self, X):
		return self._embedder(X)

	def decode(self, X):
		return self._recon(X)
