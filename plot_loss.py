"""
Python script to plot live data from training model losses.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob


path = glob('train*/')[-1]
fname = 'train_log.txt'
fig = plt.figure()

def animate(i):
	d = pd.read_csv(path+fname, sep=' ', skiprows=10).values
	epochs = np.array([val.split('=')[-1] for val in d[:,4]], dtype=float)
	lossAcc = np.array([val.split('=')[-1] for val in d[:,5]], dtype=float)
	lossFuse = np.array([val.split('=')[-1] for val in d[:,7]], dtype=float)
	plt.cla()
	plt.plot(epochs, lossAcc)
	plt.xlabel('epoch')
	plt.ylabel('Loss')
	plt.tight_layout()
	plt.savefig(path+'loss_graph.png')

ani = animation.FuncAnimation(fig, animate, 5000)
plt.show()