import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import segyio
from util import MaskGenerator, DataChunker
from pconv_model import PConvUnet

"Data reconstruction of theoretical pre_stk data with trained pconv network model"

segydata_org=[]
datapath = os.path.join(os.getcwd(),'Theoretical_prestk_data.sgy')
segy = segyio.open(datapath,ignore_geometry=True)
for i in range(0,501):
    segydata_org.append(segy.trace[i])
segydata_org=np.array(segydata_org)
segydata = ( segydata_org/np.max(abs(segydata_org)) + 1 ) / 2

mask_generator = MaskGenerator(segydata.shape[0],segydata.shape[1],rand_seed=150)
# mask = mask_generator.fix_generate_mask()
mask = mask_generator.sample()

masked_data = deepcopy(segydata)
masked_data[mask==0] = 0

model = PConvUnet( inference_only=True)
model.load(os.path.join(os.getcwd(),'pconv_model.h5'), train_bn=False)

chunker = DataChunker(128, 128, 30)

def plot_datas(data):
    vm = np.percentile(data[0], 99)

    fig = plt.figure(figsize=(10,10))
    fig.suptitle('', fontsize=20, y=0.96)

    ax = fig.add_subplot(221)
    ax.set_xlabel('Receiver number', fontsize=10, labelpad=6)
    ax.set_ylabel('Time(s)', fontsize=10, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(direction='out', top=True, right=True, width=1, length=3, colors='k', pad=2)
    ax.set_aspect('equal')
    xstr = np.arange(int(0.004 * 1720) + 1)
    x = np.linspace(0, 1720, int(0.004 * 1720) + 1)
    plt.yticks(x, xstr)
    plt.imshow(data[0].T, cmap='Greys', vmin=-1, vmax=1, aspect='auto')
    plt.title('Miss 50%', fontsize=14, pad=6)

    ax = fig.add_subplot(222)
    ax.set_xlabel('Receiver number', fontsize=10, labelpad=6)
    ax.set_ylabel('Time(s)', fontsize=10, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(direction='out', top=True, right=True, width=1, length=3, colors='k', pad=2)
    ax.set_aspect('equal')
    xstr = np.arange(int(0.004 * 1720) + 1)
    x = np.linspace(0, 1720, int(0.004 * 1720) + 1)
    plt.yticks(x, xstr)
    plt.imshow(data[1].T, cmap='Greys', vmin=-1, vmax=1, aspect='auto')
    plt.title('Original', fontsize=14, pad=6)

    ax = fig.add_subplot(223)
    ax.set_xlabel('Receiver number', fontsize=10, labelpad=6)
    ax.set_ylabel('Time(s)', fontsize=10, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(direction='out', top=True, right=True, width=1, length=3, colors='k', pad=2)
    ax.set_aspect('equal')
    xstr = np.arange(int(0.004 * 1720) + 1)
    x = np.linspace(0, 1720, int(0.004 * 1720) + 1)
    plt.yticks(x, xstr)
    plt.imshow(data[2].T, cmap='Greys', vmin=-1, vmax=1, aspect='auto')
    plt.title('Reconstruction', fontsize=14, pad=6)

    ax = fig.add_subplot(224)
    ax.set_xlabel('Receiver number', fontsize=10, labelpad=6)
    ax.set_ylabel('Time(s)', fontsize=10, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(direction='out', top=True, right=True, width=1, length=3, colors='k', pad=2)
    ax.set_aspect('equal')
    xstr = np.arange(int(0.004 * 1720) + 1)
    x = np.linspace(0, 1720, int(0.004 * 1720) + 1)
    plt.yticks(x, xstr)
    plt.imshow((data[3]).T, cmap='Greys', vmin=-1, vmax=1, aspect='auto')
    plt.title('Residual', fontsize=14, pad=6)

    plt.show()



print("Data with size: {}".format(segydata.shape))

chunked_datas = np.expand_dims(chunker.dimension_preprocess(deepcopy(masked_data)),-1)
chunked_masks =  np.expand_dims(chunker.dimension_preprocess(deepcopy(mask)),-1)
pred_datas = model.predict([chunked_datas, chunked_masks])
chunked_datas = np.squeeze(chunked_datas)
pred_datas = np.squeeze(pred_datas)
reconstructed_data = chunker.dimension_postprocess(pred_datas, segydata)

reconstructed_data  =  (reconstructed_data * 2 -1.0) *np.max(abs(segydata_org))


miss_data =  deepcopy(segydata_org)
miss_data[mask==0] = 0
plot_datas([miss_data, segydata_org, reconstructed_data,reconstructed_data-segydata_org])
