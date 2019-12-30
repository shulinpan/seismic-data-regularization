import os
from datetime import datetime
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Add
from keras.layers.merge import Concatenate
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras import backend as K


class ResNet(object):
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
            self.model= self.build_resnet()
            self.compile_resnet(self.model)
        else:
            with tf.device("/cpu:0"):
                self.model = self.build_resnet()
            self.model = multi_gpu_model(self.model, gpus=self.gpus)
            self.compile_resnet(self.model)



    def build_resnet(self):
        """
        Building network based on conv
        :param train_bn: Apply BN layer or not
        :return: Return network model and input sample matrix mask
        """

        # INPUTS
        inputs_data = Input((self.data_rows, self.data_cols, 1),name='inputs_data')


        def residual_block(input, output_channels=64, kernel_size=(3, 3), stride=(1, 1)):
            x = Conv2D(output_channels, kernel_size, padding='same', strides=stride)(input)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(output_channels, kernel_size, padding='same', strides=stride)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Add()([x, input])

            residual_block.counter += 1
            return x

        residual_block.counter = 0

        conv1=Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu')(inputs_data)
        res_block1=residual_block(conv1,output_channels=64)
        res_block2 =residual_block(res_block1, output_channels=64)
        res_block3 =residual_block(res_block2, output_channels=64)
        conv2=Conv2D(1,(3,3),strides=(1,1),padding='same')(res_block3)
        outputs=Add()([conv2,inputs_data])


        model = Model(inputs=inputs_data, outputs=outputs)


        return model

    def compile_resnet(self, model, lr=0.01):
        model.compile(
            optimizer=Adam(lr=lr),
            loss=self.loss_total()
        )


    def loss_total(self):
        """
        Creates a loss function which sums all the loss components and multiplies by their weights.
        :param mask: Sample matrix mask
        :return Calculate the total error function
        """
        def loss(y_true, y_pred):
            l2 = 1/2*K.sum(K.square(y_true-y_pred))

            return l2
        return loss



    def fit(self, generator, epochs=10, plot_callback=None, *args, **kwargs):
        """Fit the U-Net to a (datas, targets) generator
        :param generator: training generator yielding (maskes_data, original_data) tuples
        :param epochs: number of epochs to train for
        :param plot_callback: callback function taking Unet model as parameter
        """
        for _ in range(epochs):
            # Fit the model
            self.model.fit_generator(
                generator,
                epochs=self.current_epoch + 1,
                initial_epoch=self.current_epoch,
                *args, **kwargs
            )

            # Update epoch
            self.current_epoch += 1

            # After each epoch predict on test images & show them
            if plot_callback:
                plot_callback(self.model)

            # Save logfile
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

    def load(self, filepath,  lr=0.01):

        self.model= self.build_resnet()
        self.compile_resnet(self.model)
        self.model.load_weights(filepath)

    def current_weightfile(self):
        assert self.weight_filepath != None, 'Must specify location of logs'
        return self.weight_filepath + "{}_weights_{}.h5".format(self.current_epoch, self.current_timestamp())




    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2, 3])
        elif K.ndim(y_true) == 3:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

