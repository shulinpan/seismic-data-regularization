import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import segyio
from util import MaskGenerator, DataChunker
from pconv_model import PConvUnet


"Data reconstruction of theoretical post_stk data with trained pconv network model"

datapath = os.path.join(os.getcwd(),'Theoretical_poststk_data.sgy')
segydata_org = segyio.open(datapath)

segydata = []
for i in range(0,200):
    segydata.append(segydata_org.trace[i])
segydata=np.array(segydata)
segydata_org = segydata


segydata = ( segydata_org/np.max(abs(segydata_org)) + 1 ) / 2

mask_generator = MaskGenerator(segydata.shape[0],segydata.shape[1],rand_seed=300)
mask = mask_generator.sample()

masked_data = deepcopy(segydata)
masked_data[mask==0] = 0

model = PConvUnet( inference_only=True)
model.load(os.path.join(os.getcwd(),'pconv_model.h5'), train_bn=False)

chunker = DataChunker(128, 128, 30)
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


def plot_seismic(data,val=200):
    line = data / val
    inverse = False
    if inverse:
        fill = np.minimum(0, line)
    else:
        fill = np.maximum(0, line)

    for i in range(0, data.shape[0], 1):
        if np.sum(abs(line[i,:])) != 0:
            plt.plot(i + line[i, :], np.arange(data.shape[1]), "black", linewidth=0.1)
            plt.fill(i + fill[i, :], np.arange(data.shape[1]), "black", linewidth=0.1)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.set_ylim(0, data.shape[1])
    ax.set_xlim(0, data.shape[0], 1)
    ax.invert_yaxis()
    plt.show()

plot_seismic(segydata_org, val=1)
plot_seismic(miss_data, val=1)
plot_seismic(reconstructed_data, val=1)
plot_seismic(reconstructed_data-segydata_org,val=1)


