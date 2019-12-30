from keras.utils import conv_utils
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Conv2D


class PConv2D(Conv2D):
    """
    Pconv definition, overriding conv2d class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        """Adapted from original _Conv() layer of Keras
        param input_shape: list of dimensions for [data, mask]

        :param input_shape:Size of input data
        """
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        self.input_dim = input_shape[0][channel_axis]
        # Data kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='data_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # Mask kernel
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):

        """
        Use the Keras conv2d method, and essentially we have
        to do here is multiply the mask with the input X, before we apply the
        convolutions. For the mask itself, we apply convolutions with all weights
        set to 1.
        Subsequently, we set all mask values >0 to 1, and otherwise 0

        :param inputs:Input data and input sampling matrix
        :return:Data and sampling matrix after partial convolution
        """

        # Both data and mask must be supplied
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception(
                'PartialConvolution2D must be called on a list of two tensors [data, mask]. Instead got: ' + str(inputs))

        normalization = K.mean(inputs[1], axis=[1, 2], keepdims=True)
        normalization = K.repeat_elements(normalization, inputs[1].shape[1], axis=1)
        normalization = K.repeat_elements(normalization, inputs[1].shape[2], axis=2)

        # Apply convolutions to data
        data_output = K.conv2d(
            (inputs[0] * inputs[1]) / normalization, self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Apply convolutions to mask
        mask_output = K.conv2d(
            inputs[1], self.kernel_mask,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Where something happened, set 1, otherwise 0
        mask_output = K.cast(K.greater(mask_output, 0), 'float32')

        if self.use_bias:
            data_output = K.bias_add(
                data_output,
                self.bias,
                data_format=self.data_format)
        # Apply activations on the data
        if self.activation is not None:
            data_output = self.activation(data_output)
        return [data_output, mask_output]


    def compute_output_shape(self, input_shape):
        """
        Calculate the size of the output tensor
        :param input_shape:Size of input tensor
        :return:Size of the output tensor
        """
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]