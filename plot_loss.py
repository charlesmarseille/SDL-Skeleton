"""
Python script to plot live data from training model losses.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import os


path = glob('train*/')[-1]
weights = glob('ADA_LSN/weights/inception_camille/*.pth')
fname1 = 'train_log.txt'
fname2 = 'test_loss.csv'
fig = plt.figure()
if len(weights) != None:
	last_weight = weights[-1]

def animate(i):
	global last_weight
	d1 = pd.read_csv(path+fname1, sep=' ', skiprows=10).values
	epochs = np.array([val.split('=')[-1] for val in d1[:,4]], dtype=float)
	lossAcc = np.array([val.split('=')[-1] for val in d1[:,5]], dtype=float)
	lossFuse = np.array([val.split('=')[-1] for val in d1[:,7]], dtype=float)
	try:
		d2 = pd.read_csv(path+fname2, sep=',', skiprows=10).values
	except:
		pass
	weight = glob('ADA_LSN/weights/inception_camille/*.pth')[-1].split('\\')[-1][5:-4]
	if weight != last_weight:
		validate_model(int(weight))
		last_weight = weight

	plt.cla()
	plt.plot(epochs, lossAcc)
	plt.xlabel('epoch')
	plt.ylabel('Loss')
	plt.tight_layout()
	plt.savefig(path+'loss_graph.png')


def validate_model(epoch):
	os.system(f'python test_RN_exon.py -W {epoch} --C 64')


ani = animation.FuncAnimation(fig, animate, 5000)
plt.show()