import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#from models import att_r2_unet, get_coswin_model
import rasterio
from rasterio import windows
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import tifffile
import torch
from Ada_LSN.model import Network
from Ada_LSN.genotypes import geno_inception as geno
from torch.autograd import Variable
#import tensorflow as tf

#physical_devices = tf.config.list_physical_devices('GPU')
#gpu = physical_devices[:1]
#tf.config.experimental.set_memory_growth(gpu[0], True)

def merge_prediction(pred_raster, idx_i, idx_j, model_pred):
    pred = pred_raster.read(window=windows.Window(idx_j, idx_i, load_size[0], load_size[1]))[:-1, :, :]
    if len(model_pred.shape)==3:
        pred = np.moveaxis(pred, 0, -1)
        for i in range(model_pred.shape[-1]):
            model_pred[:,:,-1] = model_pred[:,:,-1] * avg_filter
    else:
        pred = pred[0]
        model_pred = model_pred * avg_filter
    pred += model_pred
    n_pred = pred_raster.read(pred_raster.count, window=windows.Window(idx_j, idx_i, load_size[0], load_size[1]))
    n_pred += avg_filter
    return pred, n_pred

def save_prediction(pred_raster, idx_i, idx_j, pred):
    for i in range(pred.shape[-1]):
        pred_raster.write(pred[:,:,i], window=windows.Window(idx_j, idx_i, load_size[0], load_size[1]), indexes=i+1)

def prepare_tta(img):
    # initiate images list
    imgs_list = [np.copy(img)]
    # rotations
    for k in [1,2,3]:
        rot_img = np.rot90(img, k=k)
        imgs_list.append(rot_img)
    return np.asarray(imgs_list)

def reverse_tta(pred):
    # initiate images list
    rot_pred_list = [np.copy(pred[0])]
    # rotations
    for k in [1,2,3]:
        rot_pred = np.rot90(pred[k], k=-k)
        rot_pred_list.append(rot_pred)
    rot_pred = np.asarray(rot_pred_list).squeeze()
    rot_pred = np.mean(rot_pred, axis=0)
    return rot_pred

# Timer
start_main_time = time.time()

# Image parameters
overlap = 0.5
load_size = (128, 128)
pred_size = (256, 256)
n_labels = 1

# Create averaging filter
avg_filter = np.zeros(load_size)
avg_filter[int(load_size[0]/4):int(3*load_size[0]/4), int(load_size[1]/4):int(3*load_size[1]/4)] = 1
avg_filter = cv2.GaussianBlur(avg_filter, (load_size[0]-1,load_size[1]-1), 0)
avg_filter = avg_filter + 0.25
avg_filter = avg_filter / avg_filter.max()
#plt.imshow(avg_filter)
#plt.show()

# Define model and dataset parameters
save_path = 'D:/cmarseille/reseaux/predictions/without_mhc/ryam/epoch_10000/'
model_name = 'sdl-skeleton'
dataset_name = 'ryam' # ryam, nordCotiere
C = 32
epoch = 10000

if dataset_name == 'ryam':
    bands_path = 'U:/Projets_2022/22-0977-MESR102-CERFO-Réseau/02_Realisation/1_Donnees/chemins_ryam/variables/'
    tiles = ['zone_1', 'zone_2', 'zone_3',]
    #bands_name = ['Hillshade', 'MHC', 'Slope', 'TRI']
    bands_name = ['Hillshade', 'Slope', 'TRI']          #no MHC
    ext = '.tif'
    #means = [22775.8, 11.3, 7.8, 0.11]
    #stds = [924.8, 6.8, 6.9, 0.097]
    means = [22775.8, 7.8, 0.11]      #no MHC
    stds = [924.8, 6.9, 0.097]         #no MHC

elif dataset_name == 'nordCotiere':
    bands_path = 'U:/Projets_2022/22-0977-MESR102-CERFO-Réseau/02_Realisation/1_Donnees/chemins_nordCotiere/Images/'
    tiles = ['5oct', '20oct',]
    bands_name = ['B02', 'B03', 'B04', 'B08']
    ext = '.jp2'
    #means = [188.322, 294.600, 286.484, 1116.616]
    #stds = [332.355, 345.118, 355.546, 690.447]
    means = [188.322, 286.484, 1116.616]        #no MHC
    stds = [332.355, 355.546, 690.447]          #no MHC    

gpu_id = 0
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.set_device(gpu_id)
torch.cuda.empty_cache()
net = Network(C, 5, [0, 1, 2, 3], geno).cuda(0).eval()
net.load_state_dict(torch.load(f'./Ada_LSN/weights/inception_camille_lr1e-6_c32_no_mhc/skel_{epoch}.pth', map_location=lambda storage, loc: storage))
net.mode = 1    # put the model in test mode (dropouts inactive)


for tile in tiles:
    # read files
    if dataset_name == 'ryam':
        arr_list = [rasterio.open(bands_path+band_name+'/'+tile+'_'+band_name+ext) for band_name in bands_name]
    elif dataset_name == 'nordCotiere':
        arr_list = [rasterio.open(bands_path+tile+'/'+band_name+'_10m'+ext) for band_name in bands_name]

    # create segmentation raster
    seg_kwds = arr_list[0].profile
    seg_kwds['driver'] = 'GTiff'
    seg_kwds['count'] = n_labels+1
    seg_kwds['dtype'] = np.float32
    seg_kwds['nodata'] = 0.0
    seg_file = rasterio.open(
        save_path+model_name+'_'+tile+'.tif',
        'w+',
        BIGTIFF=True,
        **seg_kwds
    )

    idx_i = 0
    last_i, done_i = False, False
    while not done_i:
        idx_j = 0
        last_j, done_j = False, False
        print(round(idx_i / arr_list[0].shape[0] * 100, 2))
        while not done_j:
            # check if there is data in mhc window
            win_w, win_s = rasterio.transform.xy(arr_list[0].transform, idx_i, idx_j)
            win_e, win_n = rasterio.transform.xy(arr_list[0].transform, idx_i+load_size[0], idx_j+load_size[1])
            
            # read windowed data
            var_list = []
            to_save = True
            for i in range(len(bands_name)):
                data_win = windows.from_bounds(win_w, win_n, win_e, win_s, arr_list[i].transform)
                arr = arr_list[i].read(window=data_win)[0]
                nodata_val = arr.min() if arr.min() < -10000 else arr_list[i].nodata
                if (arr==nodata_val).any() or (arr.shape != (128,128)):
                #if arr.shape != (128,128):
                    to_save = False
                    break
                var_list.append(arr)
                
            if to_save:
                # prepare data array
                img = np.asarray(var_list)
                #img = np.moveaxis(img, 0, -1)
                
                # resize data
                if load_size != pred_size:
                    img = np.array([cv2.resize(band, pred_size) for band in img])

                # standardization
                img = img.reshape(img.shape[0], pred_size[0]*pred_size[1])
                img = np.true_divide(np.subtract(img.T, means), stds)
                img = img.T.reshape(img.shape[1], pred_size[0], pred_size[1])

                # change bands order per SDL-skeleton model (sklarge_RN_exon.py line 156)
                img = img[::-1]
                
                # prepare dimensions for prediction
                img = np.expand_dims(img, 0)

                # create tensor from image
                img = Variable(torch.Tensor(img.copy()).cuda(gpu_id))
                
                # model prediction
                pred = net(img, None)[0].data[0, 0].cpu().numpy().astype(np.float32)
                
                # resize prediction
                if load_size != pred_size:
                    pred = cv2.resize(pred, load_size)
                
                # merge prediction
                pred, n_pred = merge_prediction(seg_file, idx_i, idx_j, pred)
                pred = np.dstack([pred, n_pred])
                
                # Save to prob rasters
                #print(img.min(), img.max())
                #plt.subplot(121), plt.imshow(img[0,...,0])
                #plt.subplot(122), plt.imshow(pred[:,:,0])
                #plt.show()
                save_prediction(seg_file, idx_i, idx_j, pred)
                
            # calculate new idx_j
            if last_j:
                done_j = True
            else:
                idx_j += int(load_size[1] * (1.0-overlap))
                if idx_j+load_size[1] >= arr_list[0].shape[1]:
                    idx_j = arr_list[0].shape[1] - load_size[1]
                    last_j = True
        # calculate new idx_i
        if last_i:
            done_i = True
        else:
            idx_i += int(load_size[0] * (1.0-overlap))
            if idx_i+load_size[0] >= arr_list[0].shape[0]:
                idx_i = arr_list[0].shape[0] - load_size[0]
                last_i = True

    seg_file.close()
end_time = time.time()
print(' elapsed time : ' + '{:.2f}'.format(end_time-start_main_time) + 's')