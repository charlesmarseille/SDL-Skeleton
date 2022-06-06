"""
Python script to plot live data from training model losses.
"""
import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import os


parser = argparse.ArgumentParser(description='plot_loss')
parser.add_argument('-V', '--valid', default=0, type=bool)
parser.add_argument('-C', '--C', default=64, type=int)
parser.add_argument('-F', '--folder', default=None, type=str)
parser.add_argument('-W', '--weights', default=None, type=str)
parser.add_argument('-E', '--epoch', default=None, type=int)
parser.add_argument('-T', '--title', default=None, type=str)
args = parser.parse_args()



if args.folder:
	path = args.folder+'/'
else:	
	path = glob('train*/')[-1]

if args.weights:
	weights = args.weights
else:
	weights = glob('ADA_LSN/weights/inception_camille/*.pth')

if args.epoch:
	last_weight = args.epoch
elif len(weights) != None:
	last_weight = weights[-1]

fname1 = 'train_log.txt'
fname2 = 'test_loss.csv'
fig = plt.figure()


def animate(i):
	global last_weight
	d1 = pd.read_csv(path+fname1, header=None, delim_whitespace=True, skiprows=10).values
	epochs = np.array([val.split('=')[-1] for val in d1[:,4]], dtype=float)
	lossAcc = np.array([val.split('=')[-1] for val in d1[:,5]], dtype=float)
	lossFuse = np.array([val.split('=')[-1] for val in d1[:,6]], dtype=float)
	# try:
	# 	d2 = pd.read_csv(path+fname2, sep=',', skiprows=10).values
	# except:
	# 	pass
	if args.valid:
		weight = glob('ADA_LSN/weights/inception_camille/*.pth')[-1].split('\\')[-1][5:-4]
		if (weight != last_weight):
			validate_model(int(weight))
			last_weight = weight

	plt.cla()
	plt.plot(epochs, lossAcc, label='C=64')
	plt.xlabel('epoch')
	plt.ylabel('Loss')
	plt.title(args.title)
	plt.tight_layout()
	plt.savefig(path+'loss_graph.png')


def validate_model(epoch):
	os.system(f'python test_RN_exon.py -W {epoch} --C {args.C}')


ani = animation.FuncAnimation(fig, animate, 5000)
plt.show()