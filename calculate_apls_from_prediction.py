import numpy as np
import matplotlib.pyplot as plt
import tifffile as tff
from skimage.morphology import skeletonize
import sknw
import cv2

def get_skeleton(seg):
    # smooth
    uniform_kernel_3x3 = np.ones((3,3)).astype(np.uint8)
    ellipse_kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, uniform_kernel_3x3)
    seg = cv2.dilate(seg, ellipse_kernel_3x3, iterations=1)
    seg = cv2.erode(seg, ellipse_kernel_3x3, iterations=1)
    seg = cv2.blur(seg, (5,5))
    (T, threshInv) = cv2.threshold(seg, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # skiletonize
    skel = skeletonize(seg).astype(np.uint8)
    return skel

# open predicted map and skeletonize
zone = '2'
if zone == '3':
	path_pred = f'D:/cmarseille/reseaux/predictions/ryam/with_mhc/sdl-skeleton_zone.tif'
	path_gt = f'D:/cmarseille/reseaux/chemins_ryam/correction_S2/S2_corrRaster.tif'	
else:
	path_pred = f'D:/cmarseille/reseaux/predictions/ryam/with_mhc/sdl-skeleton_zone_{zone}.tif'
	path_gt = f'D:/cmarseille/reseaux/chemins_ryam/correction_Camille/corr_seg_{zone}.tif'

pred = tff.imread(path_pred)[0]

th = 0.8

pred[pred<th] = 0
pred[pred>th] = 1
pred = pred.astype(np.uint8)
gt = tff.imread(path_gt).astype(np.uint8)


ske_pred = get_skeleton(pred)
ske_gt = get_skeleton(gt)

#ske_pred = skeletonize(pred).astype(np.uint8)
#ske_gt = skeletonize(gt).astype(np.uint8)

# build graph from skeleton
params = ['multi=False', 'iso=False', 'ring=False', 'full=False']
graph_pred = sknw.build_sknw(ske_pred, multi=True)						# multi enables multiple edges between nodes (loops are also possible)
graph_gt = sknw.build_sknw(ske_gt, multi=True)

# draw image
fig, ax = plt.subplots(1,2)
ax[0].imshow(gt, cmap='gray')
ax[1].imshow(pred, cmap='gray')

#ax[0].imshow(ske_gt, cmap='inferno')
#ax[1].imshow(ske_pred, cmap='inferno')

# draw edges by pts
for (a,b,c) in graph_gt.edges:
    ps_gt = graph_gt[a][b][c]['pts']
    ax[0].plot(ps_gt[:,1], ps_gt[:,0], 'green')

for (a,b,c) in graph_pred.edges:
	ps_pred = graph_pred[a][b][c]['pts']
	ax[1].plot(ps_pred[:,1], ps_pred[:,0], 'green')
	    
# draw node by o
nodes_gt = graph_gt.nodes()
ps_gt = np.array([nodes_gt[i]['o'] for i in nodes_gt])
ax[0].plot(ps_gt[:,1], ps_gt[:,0], 'r.')

nodes_pred = graph_pred.nodes()
ps_pred = np.array([nodes_pred[i]['o'] for i in nodes_pred])
ax[1].plot(ps_pred[:,1], ps_pred[:,0], 'r.')