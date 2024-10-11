import os
import re

import psycopg2
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Grayscale
from torch.nn import functional as F
from tqdm import tqdm

class ImageSequenceDataset(Dataset):
	def __init__(self, game=None):
		self._conn = psycopg2.connect('dbname=atariobs user=csreid')
		self._len = None
		self._rs = Resize((160,160))
		self._gs = Grayscale()
		self._idx_map = {}
		self._game = game

	def __len__(self):
		if self._len is not None:
			return self._len

		if self._game is None:
			q = "select count(*) from (select distinct episode from observation) as foo;"
		else:
			q = f"select count(*) from (select distinct episode from observation where game = '{self._game}') as foo;"
		cursor = self._conn.cursor()
		cursor.execute(q)
		count = cursor.fetchall()[0][0]
		cursor.close()
		self._len = count
		return self._len

	def __getitem__(self, idx):
		if idx not in self._idx_map:
			if self._game is None:
				ep_q = f"""
					select distinct episode from observation limit 1 offset {idx}
				"""
			else:
				ep_q = f"""
					select distinct episode from observation where game = '{self._game}' limit 1 offset {idx}
				"""
			cursor = self._conn.cursor()
			cursor.execute(ep_q)
			selected_ep = cursor.fetchall()[0][0]
			cursor.close()
			self._idx_map[idx] = selected_ep

		epid = self._idx_map[idx]
		q = f"""
			select
				action,
				state
			from
				observation
			where
				episode = '{epid}'
			order by
				step
			limit 800
		"""
		cursor = self._conn.cursor()
		cursor.execute(q)
		rows = cursor.fetchall()

		try:
			img_s = torch.stack([
				self._rs(
					self._gs(
						torch.transpose(
							torch.tensor(pickle.loads(r[1])).float(),
							0,
							2
						)
					)
				) / 255.
				for r
				in rows
			], dim=0)
		except RuntimeError as e:
			tqdm.write(f'Failed at index: {idx}')
			tqdm.write(f'\tobservation id: {obsid}')
			tqdm.write(f'\tobservation: {row}')
			raise

		action_s = torch.tensor([
			r[0] for r in rows
		])
		action_s = F.one_hot(action_s, num_classes=6)

		return (
			(img_s[:-1], action_s[:-1]), # input
			img_s[1:] # result
		)

class ImageDataset(Dataset):
	def __init__(self, game=None):
		self._conn = psycopg2.connect('dbname=atariobs user=csreid')
		self._len = None
		self._rs = Resize((160,160))
		self._gs = Grayscale()
		self.game = game

	def __len__(self):
		if self._len is not None:
			return self._len

		if self.game is None:
			q = "select count(*) from observation"
		else:
			q = f"select count(*) from observation where game = '{self.game}'"
		cursor = self._conn.cursor()
		cursor.execute(q)
		count = cursor.fetchall()[0][0]
		cursor.close()
		self._len = count
		return self._len

	def __getitem__(self, idx):
		if self.game is None:
			q = f"""
				select
					obs_id,
					state
				from observation
				limit 1
				offset {idx}
			"""
		else:
			q = f"""
				select
					obs_id,
					state
				from observation
				where game = '{self.game}'
				limit 1
				offset {idx}
			"""
		cursor = self._conn.cursor()
		cursor.execute(q)

		row = cursor.fetchall()[0]

		obsid = row[0]
		row = row[1]

		try:
			state_tensor = torch.transpose(torch.tensor(pickle.loads(row)).float(), 0, 2)
			state_tensor = self._gs(state_tensor)
			state_tensor = self._rs(state_tensor) / 255.

		except RuntimeError as e:
			tqdm.write(f'Failed at index: {idx}')
			tqdm.write(f'\tobservation id: {obsid}')
			tqdm.write(f'\tobservation: {row}')
			raise

		return state_tensor, state_tensor

if __name__ == '__main__':
	from models.vision_autoencoder import VisionAutoencoder

	isd = ImageSequenceDataset()
	loader = DataLoader(isd, 2, shuffle=True)

	imgs, actions = next(iter(loader))
	(b, s, c, h, w) = imgs.shape

	visin = VisionAutoencoder(16)
	visin.load_state_dict(torch.load('ae_grayscale_small_16.pt'))

	out_reshape = visin.embed(imgs.flatten(start_dim=0, end_dim=1)).unflatten(dim=0, sizes=(b, s))

	out_other = torch.stack([
		visin.embed(imgs[:, i])
		for i in range(s)
	], dim=1)

	print(actions.shape)
