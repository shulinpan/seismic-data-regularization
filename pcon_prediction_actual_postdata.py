import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import segyio
from util import MaskGenerator, DataChunker
from pconv_model import PConvUnet



"Data reconstruction of actual post_stk data with trained pconv network model"

datapath = os.path.join(os.getcwd(),'Actual_poststk_data.sgy')
segydata_org = segyio.open(datapath)

segydata = []
for i in range(0,230):
    segydata.append(segydata_org.trace[i])
segydata=np.array(segydata)
segydata_org = segydata
segydata = ( segydata_org/np.max(abs(segydata_org)) + 1 ) / 2

mask_generator = MaskGenerator(segydata.shape[0],segydata.shape[1],rand_seed=220)
mask = mask_generator.sample()

masked_data = deepcopy(segydata)
masked_data[mask==0] = 0


model = PConvUnet( inference_only=True)
model.load(os.path.join(os.getcwd(),'pconv_model.h5'), train_bn=False)


chunker = DataChunker(128, 128, 30)


def plot_datas(data):
    vm = np.percentile(data[0], 99)

    fig = plt.figure(figsize=(10,10))
    fig.suptitle('Inline 250', fontsize=20, y=0.96)


    ax = fig.add_subplot(221)
    ax.set_xlabel('CDP', fontsize=10, labelpad=6)
    ax.set_ylabel('Time(s)', fontsize=10, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(direction='out', top=True, right=True, width=1, length=3, colors='k', pad=2)
    ax.set_aspect('equal')
    plt.imshow(data[0].T, cmap='Greys', vmin=-4000, vmax=4000,aspect=0.3)
    plt.title('Miss 50%', fontsize=14, pad=6)


    ax = fig.add_subplot(222)
    ax.set_xlabel('CDP', fontsize=10, labelpad=6)
    ax.set_ylabel('Time(s)', fontsize=10, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(direction='out', top=True, right=True, width=1, length=3, colors='k', pad=2)
    ax.set_aspect('equal')
    plt.imshow(data[1].T, cmap='Greys', vmin=-4000, vmax=4000,aspect=0.3 )
    plt.title('Original', fontsize=14, pad=6)

    ax = fig.add_subplot(223)
    ax.set_xlabel('CDP', fontsize=10, labelpad=6)
    ax.set_ylabel('Time(s)', fontsize=10, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(direction='out', top=True, right=True, width=1, length=3, colors='k', pad=2)
    ax.set_aspect('equal')
    plt.imshow(data[2].T,cmap='Greys', vmin=-4000, vmax=4000,aspect=0.3)
    plt.title('Reconstruction', fontsize=14, pad=6)



    ax = fig.add_subplot(224)
    ax.set_xlabel('CDP', fontsize=10, labelpad=6)
    ax.set_ylabel('Time(s)', fontsize=10, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(direction='out', top=True, right=True, width=1, length=3, colors='k', pad=2)
    ax.set_aspect('equal')
    plt.imshow(data[3].T, cmap='Greys', vmin=-4000, vmax=4000,aspect=0.3 )
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
plot_datas([miss_data, segydata_org, reconstructed_data,(reconstructed_data-segydata_org)])


trace = 150#98
fig=plt.figure(figsize=(15,3))
plt.plot(np.arange(200, 1000), segydata_org[trace, 200:1000], "blue", linewidth=2,label='Original')
plt.plot(np.arange(200,1000), reconstructed_data[trace, 200:1000], "red", linewidth=2, linestyle='--' ,label='Interpolation')
plt.plot(np.arange(200, 1000), (reconstructed_data-segydata_org)[trace, 200:1000], "black",linestyle='-.', linewidth=2,label='residual')
plt.ylim(-4000,6000)
plt.legend(loc='upper right')
plt.xlabel('Time(s)', fontsize='x-large')
plt.ylabel('Amplitude', fontsize='x-large')
plt.show()

