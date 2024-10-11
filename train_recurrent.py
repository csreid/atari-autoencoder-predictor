import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss, L1Loss
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import to_pil_image, rotate, to_grayscale
from dataset import ImageSequenceDataset
from tqdm import tqdm
from models.recurrent import StatePredictor
from models.vision_autoencoder import VisionAutoencoder

dataset = ImageSequenceDataset(game='Breakout-v4')
embedder = VisionAutoencoder(128)

model = StatePredictor(
	embedder, 6
)

writer = SummaryWriter()

def _collate_fn(samples):
	imgs_in = [s[0][0] for s in samples]
	actions_in = [s[0][1] for s in samples]
	imgs_out = [s[1] for s in samples]

	padded_in = pad_sequence(imgs_in, batch_first=True)
	padded_in_a = pad_sequence(actions_in, batch_first=True)
	padded_out = pad_sequence(imgs_out, batch_first=True)
	out_shape = padded_out.shape

	# Batch size, then sequence length
	mask = torch.ones(out_shape)

	for idx in range(out_shape[0]):
		mask[idx, imgs_in[idx].shape[0]:] = 0.

	return (padded_in, padded_in_a), padded_out, mask


def log(model, dataset, loss, ctr):
	with torch.no_grad():
		model.eval()
		writer.add_scalar('loss', loss, ctr)
		loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate_fn)

		(true_video, actions), _, mask = next(iter(loader))
		est_video, h = model(true_video[:, :50].to('cuda:0'), actions[:, :50].to('cuda:0'))

		video_to_log = torch.cat(
			(
				true_video[:, :50].to('cuda:0'),
				est_video
			),
			dim=3
		)

		writer.add_video(
			'video',
			video_to_log.expand(-1, -1, 3, -1, -1),
			global_step=ctr
		)
		model.train()

def train(
	model,
	dataset,
	epochs,
	batch_size,
	output=None,
	dev='cpu',
	resume=None
):
	loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=_collate_fn)

	opt = AdamW(model.parameters())
	if resume is not None:
		try:
			model.load_state_dict(torch.load(resume))
		except:
			state = torch.load(resume)
			model.load_state_dict['model']
			opt.load_state_dict['opt']

	model = model.to(dev)

	total_prog = tqdm(range(epochs), position=0, total=epochs)
	loss_fn = MSELoss(reduction='none')
	ctr=0
	sublen = 30

	for epoch in total_prog:
		epoch_prog = tqdm(loader, total=int(len(dataset) / batch_size), position=1, leave=False)

		for (rollout_imgs, rollout_actions), next_imgs, mask in epoch_prog:
			h = None
			loss_steps = 0
			total_loss = 0.
			for idx in tqdm(range(0, rollout_imgs.shape[1], sublen), leave=False):
				img_s = rollout_imgs[:, idx:idx+sublen]
				action_s = rollout_actions[:, idx:idx+sublen]
				Y = next_imgs[:, idx:idx+sublen]
				submask = mask[:, idx:idx+sublen].to(dev)

				img_s, action_s = (img_s.to(dev), action_s.to(dev))
				Y = Y.to(dev)
		
				Y_pred, h = model(img_s, action_s)

				Y_pred = Y_pred.to(dev)

				loss = loss_fn(Y_pred, Y)
				loss = loss * submask
				loss = loss.sum() / submask.sum()

				opt.zero_grad()
				loss.backward()
				opt.step()

				ctr += 1
				with torch.no_grad():
					total_loss += loss
					loss_steps += 1
				h = (hi.detach() for hi in h)

			log(model, dataset, total_loss / loss_steps, ctr)
			if output and (ctr % 10) == 0:
				state = {
					"model": model.state_dict(),
					"opt": opt.state_dict()
				}
				torch.save(model.state_dict(), output)


	return model

if __name__ == '__main__':
	train(
		model,
		dataset,
		epochs=200,
		batch_size=4,
		dev='cuda:0',
		#output='recurrent5.pt',
	)
