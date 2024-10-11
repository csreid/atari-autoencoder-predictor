import zmq
from dataclasses import dataclass
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

@dataclass
class StateMsg:
	img: list[list[float]]
	action: int | float
	reward: float

class PublishCallback(BaseCallback):
	def __init__(self, pub_addr):
		self.ctx = zmq.Context()
		self.sock = self.ctx.socket(zmq.PUSH)
		self.addr = pub_addr

	def send(self, data: StateMsg):
		return self.sock.send(bytes(json.dumps({**data}), 'utf-8'))
