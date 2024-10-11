from dotenv import load_dotenv
load_dotenv()
import pickle
import os
import torch as th
import zmq
import psycopg2
from logrecord import LogRecord

dbstring = os.getenv('DBSTRING')
conn = psycopg2.connect(dbstring)

ctx = zmq.Context()
sock = ctx.socket(zmq.PULL)
sock.bind('tcp://127.0.0.1:5555')


def _save_data(data):
	obj = pickle.loads(data)
	cur = conn.cursor()
	cur.execute("""
		insert into observation (
			state,
			action,
			reward,
			done,
			step,
			episode,
			game
		) values (
			%s, %s, %s, %s, %s, %s, %s
		)
	""", (
		pickle.dumps(obj.state),
		obj.action,
		obj.reward,
		obj.done,
		obj.step,
		str(obj.ep),
		obj.game
	))

	conn.commit()

while True:
	data = sock.recv()
	_save_data(data)

