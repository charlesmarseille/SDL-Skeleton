from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib

# Load fnames
fns = glob('*.jpg')
fns_rot_mask = np.array([fn.find('_rot') for fn in fns])>=0
fns_rot = np.array(fns)[fns_rot_mask]
fns_no_rot = np.array(fns)[~fns_rot_mask]

# Load images and compute stats
ims = [cv2.imread(fn, 0) for fn in fns]
mins = np.array([np.min(im) for im in ims])
maxs = np.array([np.max(im) for im in ims])
stds = np.array([np.std(im) for im in ims])
means = np.array([np.mean(im) for im in ims])

# Plot stats
plt.figure()
plt.hist(means)
plt.figure()
plt.hist(stds)
plt.title('std')
plt.figure()
plt.hist(mins)
plt.title('min')
plt.figure()
plt.hist(maxs)
plt.title('max')
