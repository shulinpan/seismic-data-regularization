import numpy as np
import os
import gc
from copy import deepcopy



class AugmentingDataGenerator(object):
    """
    Read data dynamically to reduce memory
    """
    def __init__(self,data_path,rows=128,cols=128,n_channels=1, batch_size=4, shuffle=True):
        """
        :param data_path: File path.
        :param rows: Data height.
        :param cols: Data width.
        :param n_channels: channels of data.
        :param batch_size: Batch size of data.
        :param shuffle: Whether to scramble data.
        """
        self.data_path = data_path
        self.cols = cols
        self.rows = rows
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.data_list=os.listdir(self.data_path)
        self.data_list.sort(key=lambda x: int(x[:-4]))   #os.listdir不是按顺序排列，x[:-?]选择文件名第后?位排序
        self.shuffle = shuffle
        self.on_epoch_end()
        self.index = 1

    def __len__(self):
        "Count the number of batches per epoch"
        return int(np.floor(len(self.data_list)  / self.batch_size))

    def __getitem__(self):
        "Read the data of each batch"
        data = np.empty((self.batch_size, self.rows, self.cols, 1))
        for i in range(self.batch_size):
            data_tmp = self.copy_data_list.pop(0)  # 按顺序提取文件
            data[i] = np.expand_dims(np.load(os.path.join(self.data_path, data_tmp)), -1)
            data[i] = ( data[i]/np.max(abs(data[i])) + 1 ) / 2
        return data


    def on_epoch_end(self):
        "Scrambling data after each epoch"
        self.copy_data_list = deepcopy(self.data_list)
        if self.shuffle == True:
            np.random.shuffle(self.copy_data_list)

    def flow_from_directory(self,mask_generator,seed = None):
        """
        :param mask_generator:Instantiation object of sampling matrix mask
        :param seed:Random number seed
        :return:Iterator of sampling matrix mask
        """
        while True:
            if self.index <= self.__len__():
                ori = self.__getitem__()
            else:
                self.on_epoch_end()
                ori = self.__getitem__()
                self.index = 1

            mask = np.stack([
                mask_generator.sample(seed)
                for _ in range(ori.shape[0])], axis=0
            )
            masked = deepcopy(ori)
            masked[mask == 0] = 0

            mask = np.expand_dims(mask,-1)

            self.index +=1

            gc.collect()
            yield [masked, mask], ori