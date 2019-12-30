import os
from datetime import datetime
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras import backend as K
from pconv_layer import PConv2D

class PConvUnet(object):
    def __init__(self, data_rows=128, data_cols=128, weight_filepath=None,inference_only=False, net_name='default', gpus=1):
        """Create the PConvUnet. If variable data size, set data_rows and data_cols to None
        :param data_rows (int): data height.
        :param data_cols (int): data width.
        :param inference_only (bool): initialize BN layers for inference.
        :param net_name (str): Name of this network (used in logging).
        :param gpus (int): How many GPUs to use for training.
        """
        self.weight_filepath = weight_filepath
        self.data_rows = data_rows
        self.data_cols = data_cols
        self.img_overlap = 30
        self.inference_only = inference_only
        self.net_name = net_name
        self.gpus = gpus

        assert self.data_rows >= 64, 'Height must be >64 '
        assert self.data_cols >= 64, 'Width must be >64 '

        self.current_epoch = 0

        if self.gpus <= 1:
            self.model, inputs_mask = self.build_pconv_unet()
            self.compile_pconv_unet(self.model, inputs_mask)
        else:
            with tf.device("/cpu:0"):
                self.model, inputs_mask = self.build_pconv_unet()
            self.model = multi_gpu_model(self.model, gpus=self.gpus)
            self.compile_pconv_unet(self.model, inputs_mask)



    def build_pconv_unet(self, train_bn=True):
        """
        Building u-net network based on pconv
        :param train_bn: Apply BN layer or not
        :return: Return network model and input sample matrix mask
        """
        # INPUTS
        inputs_data = Input((self.data_rows, self.data_cols, 1),name='inputs_data')
        inputs_mask = Input((self.data_rows, self.data_cols, 1),name='inputs_mask')

        # ENCODER
        def encoder_layer(data_in, mask_in, filters, kernel_size, bn=True):
            conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([data_in, mask_in])
            if bn:
                conv = BatchNormalization(name='EncBN' + str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation('relu')(conv)
            encoder_layer.counter += 1
            return conv, mask

        encoder_layer.counter = 0

        e_conv1, e_mask1 = encoder_layer(inputs_data, inputs_mask, 32, 7, bn=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 64, 5)
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 128, 5)
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 256, 3)
        e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3)
        e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 512, 3)

        # DECODER
        def decoder_layer(data_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
            up_data = UpSampling2D(size=(2, 2))(data_in)
            up_mask = UpSampling2D(size=(2, 2))(mask_in)
            concat_data = Concatenate(axis=3)([e_conv, up_data])
            concat_mask = Concatenate(axis=3)([e_mask, up_mask])
            conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_data, concat_mask])
            if bn:
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
            return conv, mask


        d_conv7, d_mask7 = decoder_layer(e_conv6, e_mask6, e_conv5, e_mask5, 512, 3)
        d_conv8, d_mask8 = decoder_layer(d_conv7, d_mask7, e_conv4, e_mask4, 256, 3)
        d_conv9, d_mask9 = decoder_layer(d_conv8, d_mask8, e_conv3, e_mask3, 128, 3)
        d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv2, e_mask2, 64, 3)
        d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv1, e_mask1, 32, 3)
        d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, inputs_data, inputs_mask, 1, 3, bn=False)
        outputs = Conv2D(1, 1, activation='sigmoid')(d_conv12)

        model = Model(inputs=[inputs_data, inputs_mask], outputs=outputs)

        return model,inputs_mask


    def compile_pconv_unet(self, model, inputs_mask, lr=0.0002):
        model.compile(
            optimizer=Adam(lr=lr),
            loss=self.loss_total(inputs_mask)
        )


    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components and multiplies by their weights.
        :param mask: Sample matrix mask
        :return Calculate the total error function
        """
        def loss(y_true, y_pred):
            y_comp = mask * y_true + (1-mask) * y_pred

            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            l3 = self.loss_tv(mask, y_comp)

            return l1 + 6*l2 +  0.1*l3
        return loss

    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)


    def loss_tv(self, mask, y_comp):
        """Total variation loss"""
        kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp
        a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])
        b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])
        return a+b

    def fit(self, generator, epochs=10, plot_callback=None, *args, **kwargs):
        """Fit the U-Net to a (datas, targets) generator
        :param generator: training generator yielding (maskes_data, original_data) tuples
        :param epochs: number of epochs to train for
        :param plot_callback: callback function taking Unet model as parameter
        """
        for _ in range(epochs):
            self.model.fit_generator(
                generator,
                epochs=self.current_epoch + 1,
                initial_epoch=self.current_epoch,
                *args, **kwargs
            )

            self.current_epoch += 1

            if plot_callback:
                plot_callback(self.model)

            if self.weight_filepath:
                self.save()

    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)

    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def save(self):
        self.model.save_weights(self.current_weightfile())

    def load(self, filepath, train_bn=True, lr=0.0002):

        self.model,inputs_mask = self.build_pconv_unet(train_bn)
        self.compile_pconv_unet(self.model, inputs_mask, lr)
        self.model.load_weights(filepath)

    def current_weightfile(self):
        assert self.weight_filepath != None, 'Must specify location of logs'
        return self.weight_filepath + "{}_weights_{}.h5".format(self.current_epoch, self.current_timestamp())




    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    @staticmethod
    def l1(y_true, y_pred):
        """
        Calculate the L1 loss used in all loss calculations
        :param y_true: Complete data
        :param y_pred: Reconstruction data
        :return: L1 error of complete data and reconstructed data
        """
        if K.ndim(y_true) == 4:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2, 3])
        elif K.ndim(y_true) == 3:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")
