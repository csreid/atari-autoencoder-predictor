import torch as th
from dataclasses import dataclass

@dataclass
class LogRecord:
	state: th.Tensor
	action: th.Tensor
	reward: float
	done: bool
	step: int
	ep: int
	game: str
