import torch
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import LSTM, Linear, functional as F, Sequential, ReLU, Sigmoid, BatchNorm1d, BatchNorm2d, Module

class StatePredictor(Module):
	def __init__(self, img_embedder, n_actions):
		super().__init__()

		self._img_embedder = img_embedder
		img_features = self._img_embedder._n_features

		self._rnn = LSTM(
			input_size=img_features+n_actions,
			hidden_size=img_features,
			num_layers=1,
			batch_first=True
		)

		#self._fc = Linear(64, img_features)

	def embed(self, imgs):
		if len(imgs.shape) == 5:
			(b, s, c, h, w) = imgs.shape
			return self._img_embedder.embed(
				imgs.flatten(start_dim=0, end_dim=1)
			).unflatten(dim=0, sizes=(b, s))

		return self._img_embedder.embed(imgs)

	def forward(self, imgs, actions, hidden=None):
		(b, s, c, h, w) = imgs.shape
		embs = self.embed(imgs)
		out = torch.cat((embs, actions), dim=2)

		out, hidden = self._rnn(out, hidden)
		out = F.relu(out)
		#out = self._fc(out)
		out = self._img_embedder.decode(out)
		out = out.reshape(b, s, c, h, w)

		return out, hidden
