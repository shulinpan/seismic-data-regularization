from random import randint, seed
import numpy as np
import cv2
import random

class MaskGenerator():
    def __init__(self, width, height,  rand_seed=None, filepath=None):
        """Convenience functions for generating masks to be used for training

        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width

        Keyword Arguments:
            rand_seed {[type]} -- Random seed (default: {None})

        """
        self.height = height
        self.width = width

        if rand_seed:
            seed(rand_seed)

    def _generate_mask(self):
        """
        Generate mask with less than 50% missing ratio

        Output:sample matrix mask
        """
        mask = np.zeros((self.width, self.height), np.uint8)
        size = int((self.width + self.height) * 0.01)
        if self.width < 32 or self.height < 32:
            raise Exception("Width and Height of mask must be at least 64!")
        for _ in range(randint(1,int(0.5*self.width))):
            x1 = randint(0, self.width-1)
            thickness = 1
            cv2.line(mask, (0, x1),(self.height-1, x1), 1, thickness)
        return 1 - mask

    def fix_generate_mask(self):
        """
        Generate mask with fixed missing scale

        Output:sample matrix mask
        """
        mask = np.zeros((self.width, self.height), np.uint8)
        size = int((self.width + self.height) * 0.01)
        if self.width < 32 or self.height < 32:
            raise Exception("Width and Height of mask must be at least 64!")
        i = 0
        for _ in range(1, int(0.3 * self.width)):
            index_xline = random.sample(range(0, self.width), int(0.5 * self.width))
            x1 = index_xline[i]
            i += 1
            thickness = 1
            cv2.line(mask, (0, x1), (self.height - 1, x1), 1, thickness)

        return 1 - mask


    def sample(self, random_seed=None):
        """Retrieve a random mask"""
        if random_seed:
            seed(random_seed)
        return self._generate_mask()

class DataChunker(object):

    def __init__(self, rows, cols, overlap):
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
    def perform_chunking(self, data_size, chunk_size):

        """
        Given an data dimension data_size, return list of (start, stop)
        tuples to perform chunking of chunk_size
        """

        chunks, i = [], 0
        while True:
            chunks.append((i * (chunk_size - self.overlap / 2), i * (chunk_size - self.overlap / 2) + chunk_size))
            i += 1
            if chunks[-1][1] > data_size:
                break

        n_count = len(chunks)
        chunks[-1] = tuple(x - (n_count * chunk_size - data_size - (n_count - 1) * self.overlap / 2) for x in chunks[-1])
        chunks = [(int(x), int(y)) for x, y in chunks]
        return chunks

    def get_chunks(self, data, scale=1):

        """
        Get width and height lists of (start, stop) tuples for chunking of data.
        """
        x_chunks, y_chunks = [(0, self.rows)], [(0, self.cols)]
        if data.shape[0] > self.rows:
            x_chunks = self.perform_chunking(data.shape[0], self.rows)
        else:
            x_chunks = [(0, data.shape[0])]
        if data.shape[1] > self.cols:
            y_chunks = self.perform_chunking(data.shape[1], self.cols)
        else:
            y_chunks = [(0, data.shape[1])]
        return x_chunks, y_chunks

    def dimension_preprocess(self, data, padding=True):
        """
        In case of prediction on data of different size than 128x128,
        this function is used to add padding and chunk up the data into pieces
        of 128x128, which can then later be reconstructed into the original data
        using the dimension_postprocess() function.
        """

        assert len(data.shape) == 2, "Data dimension expected to be ( xline, samp_point)"
        if padding:
            if data.shape[0] < self.rows:
                padding = np.ones((self.rows - data.shape[0], data.shape[1]))
                data = np.concatenate((data, padding), axis=0)
            if data.shape[1] < self.cols:
                padding = np.ones((data.shape[0], self.cols - data.shape[1]))
                data = np.concatenate((data, padding), axis=1)
        x_chunks, y_chunks = self.get_chunks(data)
        images = []
        for x in x_chunks:
            for y in y_chunks:
                images.append(
                    data[x[0]:x[1], y[0]:y[1]]
                )
        images = np.array(images)

        return images

    def dimension_postprocess(self, chunked_data, original_data, scale=1, padding=True):
        """
        In case of prediction on data of different size than 128x128,
        the dimension_preprocess  function is used to add padding and chunk
        up the data into pieces of 128x128, and this function is used to
        reconstruct these pieces into the original data.
        """

        assert len(original_data.shape) == 2, "data dimension expected to be (xline ,samp_point)"
        assert len(chunked_data.shape) == 3, "Chunked data dimension expected to be (batch_size, xline, samp_point)"

        if padding:
            if original_data.shape[0] < self.rows:
                new_images = []
                for data in chunked_data:
                    new_images.append(data[0:scale * original_data.shape[0], :])
                    chunked_data = np.array(new_images)

            if original_data.shape[1] < self.cols:
                new_images = []
                for data in chunked_data:
                    new_images.append(data[:, 0:scale * original_data.shape[1]])
                    chunked_data = np.array(new_images)

        new_shape = (
            original_data.shape[0] * scale,
            original_data.shape[1] * scale
        )
        reconstruction = np.zeros(new_shape)
        x_chunks, y_chunks = self.get_chunks(original_data)

        i = 0
        s = scale
        for x in x_chunks:
            for y in y_chunks:
                prior_fill = reconstruction != 0
                chunk = np.zeros(new_shape)
                chunk[x[0] * s:x[1] * s, y[0] * s:y[1] * s] += chunked_data[i]
                chunk_fill = chunk != 0
                reconstruction += chunk
                reconstruction[prior_fill & chunk_fill] = reconstruction[prior_fill & chunk_fill] / 2
                i += 1
        return reconstruction


