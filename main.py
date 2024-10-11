from dotenv import load_dotenv
load_dotenv()
import uuid
import os
import torch as th
import pickle
import gymnasium as gym
from gymnasium.wrappers import FrameStack, NormalizeReward
from gymnasium import RewardWrapper
from dataclasses import dataclass
import zmq
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from dataclasses import dataclass
from logrecord import LogRecord

class ObservableEnv(gym.Env):
	def __init__(self, env):
#		super().__init__(env)
		self.env = env
		self.observation_space = env.observation_space
		self.action_space = env.action_space
		self._state = None
		self._previous_state = None
		self._action = 0
		self._reward = 0
		self._is_done = False
		self._steps = None
		self._episode_id = None

	@property
	def episode_id(self):
		if self._episode_id is None:
			raise Exception(f'No steps have been taken yet')

		return self._episode_id

	@property
	def steps(self):
		if self._steps is None:
			raise Exception(f'No steps have been taken yet')

		return self._steps

	@property
	def state(self):
		if self._state is None:
			raise Exception(f'No steps have been taken yet')

		return self._obs

	@property
	def action(self):
		return self._action

	@property
	def reward(self):
		if self._reward is None:
			raise Exception(f'No steps have been taken yet')

		return self._reward

	@property
	def is_done(self):
		if self._is_done is None:
			raise Exception(f'No steps have been taken yet')

		return self._is_done

	def step(self, a):
		s, r, done, trunc, info = self.env.step(a)
		self._previous_state = self._state
		self._state = s
		self._action = a
		self._reward = r
		self._is_done = done

		self._steps += 1

		return (s, r, done, trunc, info)

	def reset(self, seed=None):
		self._steps = 0
		self._episode_id = uuid.uuid4()
		self._is_done = False
		self._reward = 0
		self._action = 0

		retval = self.env.reset()
		self._state=retval[0]
		self._previous_state = self._state

		return retval

	def render(self):
		return self.env.render()

	def close(self):
		return self.env.close()

	def seed(self, seed=None):
		return self.env.seed(seed)

class LogCallback(BaseCallback):
	def __init__(self, ctx, env_name):
		self.ep = 0
		self.ep_step = 0
		self.log = []

		sock = ctx.socket(zmq.PUSH)
		sock.connect('tcp://127.0.0.1:5555')
		self.sock = sock
		self._env_name = env_name

		super().__init__()

	def _send(self, data):
		return self.sock.send(pickle.dumps(data))

	def _on_step(self):
		env = self.training_env

		obs = env.envs[0].unwrapped._previous_state
		action = env.envs[0].unwrapped.action
		reward = env.envs[0].unwrapped.reward
		is_done = env.envs[0].unwrapped.is_done
		steps = env.envs[0].unwrapped.steps
		ep = env.envs[0].unwrapped.episode_id

		rec = LogRecord(
			state=obs,
			action=int(action),
			reward=float(reward),
			done=bool(is_done),
			step=steps,
			ep=ep,
			game=self._env_name
		)

		self.ep_step += 1

		self._send(rec)
		return True

class StableReward(RewardWrapper):
	def reward(self, r):
		if r > 0:
			return 1.
		if r < 0:
			return -1

		return 0

env_name = "SpaceInvaders-v4"
render = False
env = ObservableEnv(
	StableReward(
		gym.make(env_name,),
	)
)
model = DQN("MlpPolicy", env, verbose=1, buffer_size=1000, tensorboard_log='runs/')
#visenv = gym.make(env_name, render_mode="human")

ctx = zmq.Context()
callbacks = CallbackList([
	LogCallback(ctx, env_name)
])
model.learn(
	total_timesteps=1000000,
	log_interval=10,
	progress_bar=True,
	callback=callbacks,
)

print(f'Done!')
