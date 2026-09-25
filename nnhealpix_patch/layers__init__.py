# -*- encoding: utf-8 -*-

import numpy as np
import os.path
from tensorflow.keras.layers import Conv1D
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import nnhealpix as nnh


class OrderMap(Layer):
    """Defines a Keras layer able to operate on HEALPix maps.

    This layer has two purposes:

    * It reorders the input map so that neighbour pixels are all adjacent;
    * It performs 1D convolution using whatever the backend offers.
    """

    def __init__(self, indices, **kwargs):
        self.input_indices = np.array(indices, dtype="int32")
        Kindices = K.variable(indices, dtype="int32")
        self.indices = K.stop_gradient(Kindices)
        super(OrderMap, self).__init__(**kwargs)

    def build(self, input_shape):
        """Create the weights for the layer"""
        self.in_shape = input_shape
        self.output_dim = (input_shape[0], self.indices.shape[0], input_shape[2])
        super(OrderMap, self).build(input_shape)

    def call(self, x):
        """Implement the layer's logic"""
        # x = tf.to_float(x)
        x = tf.cast(x, dtype=tf.float32)
        zero = tf.fill([tf.shape(x)[0], 1, tf.shape(x)[2]], 0.0)
        x1 = tf.concat([x, zero], axis=1)
        reordered = tf.gather(x1, self.indices, axis=1)
        return reordered

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.output_dim)

    def get_config(self):
        "Return a dictionary containing the configuration for the layer."
        config = super(OrderMap, self).get_config()
        config.update({"indices": self.input_indices})
        return config


def Dgrade(nside_in, nside_out):
    """Keras layer performing a downgrade of input maps

    Parameters
    ----------
    nside_in : integer
        Nside parameter for the input maps.
        Must be a valid healpix Nside value
    nside_out: integer
        Nside parameter for the output maps.
        Must be a valid healpix Nside value
    """

    file_in = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ancillary_files",
        "dgrade_from{}_to{}.npy".format(nside_in, nside_out),
    )
    try:
        pixel_indices = np.load(file_in)
    except:
        pixel_indices = nnh.dgrade(nside_in, nside_out)

    def f(x):
        y = OrderMap(pixel_indices)(x)
        pool_size = int((nside_in / nside_out) ** 2.0)
        y = tf.keras.layers.AveragePooling1D(pool_size=pool_size)(y)
        return y

    return f


def Pooling(nside_in, nside_out, layer1D, *args, **kwargs):
    """Keras layer performing a downgrade+custom pooling of input maps

    Args:
        * nside_in (integer): ``NSIDE`` parameter for the input maps.
        * nside_out (integer): ``NSIDE`` parameter for the output maps.
        * layer1D (layer object): a 1-D layer operation, like
          :code:`keras.layers.MaxPooling1D`
        * args (any): Positional arguments to be passed to :code:`layer1D`
        * kwargs: keyword arguments to be passed to
          :code:`layer1D`. The keyword :code:`pool_size` should not be
          included, as it is handled automatically.
    """

    file_in = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ancillary_files",
        "dgrade_from{}_to{}.npy".format(nside_in, nside_out),
    )
    try:
        pixel_indices = np.load(file_in)
    except:
        pixel_indices = nnh.dgrade(nside_in, nside_out)

    def f(x):
        y = OrderMap(pixel_indices)(x)
        pool_size = int((nside_in / nside_out) ** 2.0)
        kwargs["pool_size"] = pool_size
        y = layer1D(*args, **kwargs)(y)
        return y

    return f


def MaxPooling(nside_in, nside_out):
    """Keras layer performing a downgrading+maxpooling of input maps

    Args:
        * nside_in (integer): ``NSIDE`` parameter for the input maps.
        * nside_out (integer): ``NSIDE`` parameter for the output maps.
    """

    return Pooling(nside_in, nside_out, tf.keras.layers.MaxPooling1D)


def AveragePooling(nside_in, nside_out):
    """Keras layer performing a downgrading+averaging of input maps

    Args:
        * nside_in (integer): ``NSIDE`` parameter for the input maps.
        * nside_out (integer): ``NSIDE`` parameter for the output maps.
    """

    return Pooling(nside_in, nside_out, tf.keras.layers.AveragePooling1D)


def DegradeAndConvNeighbours(
        nside_in, nside_out, filters, use_bias=False, trainable=True
):
    """Keras layer performing a downgrading and convolution of input maps.

    Args:
        * nside_in (integer): ``NSIDE`` parameter for the input maps.
        * nside_out (integer): ``NSIDE`` parameter for the output maps.
        * filters (integer): Number of filters to use in the
          convolution
        * use_bias (bool): Whether the layer uses a bias vector or
          not. Default is ``False``.
        * trainable (bool): Wheter this is a trainable layer or
          not. Default is ``True``.

    """

    file_in = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ancillary_files",
        "dgrade_from{}_to{}.npy".format(nside_in, nside_out),
    )
    try:
        pixel_indices = np.load(file_in)
    except:
        pixel_indices = nnh.dgrade(nside_in, nside_out)

    def f(x):
        y = OrderMap(pixel_indices)(x)
        kernel_size = int((nside_in / nside_out) ** 2.0)
        y = Conv1D(
            filters,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=use_bias,
            trainable=trainable,
            kernel_initializer="random_uniform",
        )(y)
        return y

    return f


def ConvNeighbours(nside, kernel_size, filters, use_bias=False, trainable=True):
    """Keras layer to perform pixel neighbour convolution.

    Args:
        * nside(integer): ``NSIDE`` parameter for the input maps.
        * kernel_size(integer): Dimension of the kernel. Currently,
          NNhealpix only supports ``kernelsize = 9`` (first-order
          convolution).
        * filters(integer): Number of filters to use in the
          convolution
        * use_bias(bool): Whether the layer uses a bias vector or
          not. Default is ``False``.
        * trainable(bool): Wheter this is a trainable layer or
          not. Default is ``True``.

    """

    if kernel_size != 9:
        raise ValueError("kernel size must be 9")

    file_in = nnh.filter_file_name(nside, kernel_size)
    try:
        pixel_indices = np.load(file_in)
    except:
        pixel_indices = nnh.filter(nside)

    def f(x):
        y = OrderMap(pixel_indices)(x)
        y = Conv1D(
            filters,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=use_bias,
            trainable=trainable,
        )(y)
        return y

    return f

class MaskedAveragePoolingLayer(Layer):
    """Keras Layer: downgrade + masked average pooling.

    The layer:
      * reorders pixels using the same OrderMap indices,
      * groups pixels into blocks of `pool_size`,
      * computes the sum of non-masked entries and divides by the number
        of non-masked entries (ignoring `mask_value`), and
      * if all entries in a block are masked, returns `mask_value`.
    """

    def __init__(self, pixel_indices, pool_size, mask_value=0.0, **kwargs):
        super(MaskedAveragePoolingLayer, self).__init__(**kwargs)
        # keep a numpy copy for get_config / debugging
        self.input_indices = np.array(pixel_indices, dtype="int32")
        # reuse OrderMap (it will create its own internal Kindices variable)
        self.ordermap = OrderMap(self.input_indices)
        self.pool_size = int(pool_size)
        self.mask_value = float(mask_value)

    def build(self, input_shape):
        # output dimension: (batch, n_pix_out, channels)
        n_pix_out = int(self.input_indices.shape[0] // self.pool_size)
        self.output_dim = (input_shape[0], n_pix_out, input_shape[2])
        super(MaskedAveragePoolingLayer, self).build(input_shape)

    def call(self, x):
        # ensure float type
        x = tf.cast(x, dtype=tf.float32)

        # reorder using the OrderMap layer
        y = self.ordermap(x)  # shape: (batch, n_pix_out * pool_size, channels)

        # dynamic shapes (symbolic-safe because we're inside Layer.call)
        batch = tf.shape(y)[0]
        n_pix_out = tf.math.floordiv(tf.shape(y)[1], self.pool_size)
        n_chan = tf.shape(y)[2]

        # reshape into groups: (batch, n_pix_out, pool_size, channels)
        y = tf.reshape(y, [batch, n_pix_out, self.pool_size, n_chan])

        # build mask of valid entries (not equal to mask_value)
        mv = tf.cast(self.mask_value, y.dtype)
        mask = tf.not_equal(y, mv)
        mask = tf.cast(mask, y.dtype)  # float mask for sums/counts

        # sum valid entries and count them (per channel)
        summed = tf.reduce_sum(y * mask, axis=2)   # shape: (batch, n_pix_out, channels)
        count = tf.reduce_sum(mask, axis=2)        # shape: (batch, n_pix_out, channels)

        # safe divide: if count==0 -> divide_no_nan yields 0, then we'll replace with mask_value
        avg = tf.math.divide_no_nan(summed, count)

        # where all masked (count == 0), set value to mask_value
        avg = tf.where(tf.greater(count, 0), avg, tf.fill(tf.shape(avg), mv))

        return avg

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.output_dim)

    def get_config(self):
        config = super(MaskedAveragePoolingLayer, self).get_config()
        # store indices as list so it's JSON-serializable like OrderMap does
        config.update(
            {
                "indices": self.input_indices,
                "pool_size": self.pool_size,
                "mask_value": self.mask_value,
            }
        )
        return config


def MaskedAveragePooling(nside_in, nside_out, mask_value=0.0):
    """Factory: returns a MaskedAveragePoolingLayer instance.

    Usage matches the earlier API:
        x = nnhealpix.layers.MaskedAveragePooling(nside_in, nside_out)(x_input)
    """
    file_in = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ancillary_files",
        "dgrade_from{}_to{}.npy".format(nside_in, nside_out),
    )
    try:
        pixel_indices = np.load(file_in)
    except Exception:
        pixel_indices = nnh.dgrade(nside_in, nside_out)

    pool_size = int((nside_in / nside_out) ** 2.0)
    return MaskedAveragePoolingLayer(pixel_indices, pool_size, mask_value=mask_value)