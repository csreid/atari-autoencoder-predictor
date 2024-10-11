import pickle
import torch as th
import zmq
from logrecord import LogRecord

ctx = zmq.Context()
sock = ctx.socket(zmq.PULL)
sock.bind('tcp://127.0.0.1:5555')

while True:
	data = sock.recv()
	print(pickle.loads(data).reward)
