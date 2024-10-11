import torch as th
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image, rotate, resize
from dataset import ImageDataset
from tqdm import tqdm
from models.vision_autoencoder import VisionAutoencoder

ds = ImageDataset()
img_features = 1024
model = VisionAutoencoder(img_features)
writer = SummaryWriter()
sample_loader = DataLoader(ds, batch_size=64, shuffle=True)
sample_set, _ = next(iter(sample_loader))

def log(model, loss, coef_loss, dataset, ctr, emb):
	with th.no_grad():
		model.eval()
		loader = DataLoader(dataset, batch_size=1, shuffle=True)
		imgs, _ = next(iter(loader))

		imgs = imgs.to('cuda:0')
		recon_imgs = model(imgs.to('cuda:0'))
		recon_mins = th.min(recon_imgs)
		recon_maxs = th.max(recon_imgs)

		new_recon = (recon_imgs - recon_mins) / (recon_maxs - recon_mins)

		imgs = th.cat(
			(
				resize(rotate(imgs, angle=-90.), (512,512)),
				resize(rotate(recon_imgs, angle=-90.), (512,512)),
				resize(rotate(new_recon, angle=-90.), (512,512)),
			),
			dim=3
		)

		writer.add_images('Reconstructed Images', imgs, ctr)
		writer.add_scalar('loss/reconstruction', loss, ctr)
		writer.add_scalar('loss/correlation', coef_loss, ctr)

		writer.add_scalar('embedding/mean', emb.mean(), ctr)
		writer.add_scalar('embedding/min', emb.min(), ctr)
		writer.add_scalar('embedding/max', emb.max(), ctr)

		writer.add_histogram('embedding/values', values=emb, global_step=ctr)
		model.train()

def train(model, dataset, epochs, batch_size, output=None, dev='cpu', resume=None):
	loader = DataLoader(dataset, batch_size, shuffle=True)

	if resume is not None:
		model.load_state_dict(th.load(resume))
	model = model.to(dev)

	opt = Adam(model.parameters(), lr=5e-2)
	loss_fn = L1Loss()
	ctr = 0

	total_prog = tqdm(range(epochs), position=0, total=epochs)
	for epoch in total_prog:
		epoch_prog = tqdm(loader, total=int(len(dataset) / batch_size), position=1, leave=False)

		for X, Y in epoch_prog:
			X = X.to(dev)
			Y = Y.to(dev)

			Y_pred, emb = model(X, return_embedding=True)
			Y_pred = Y_pred.to(dev)
			#coef = th.cov(emb)
			#coef_loss = th.abs(coef).mean()
			coef_loss = 0

			base_loss = loss_fn(Y_pred, Y)

			loss = base_loss + coef_loss

			opt.zero_grad()
			loss.backward()
			opt.step()

			ctr += 1
			if output and (ctr % 100) == 0:
				th.save(model.state_dict(), output)

			log(model, base_loss, coef_loss, dataset, ctr, emb)

	return model

train(
	model,
	ds,
	10,
	16,
	dev='cuda:0',
	output='ae_grayscale_deep_64_no_maxpool_no_coefloss_tanh.pt',
	resume='ae_grayscale_deep_64_no_maxpool_no_coefloss_tanh.pt',
)
