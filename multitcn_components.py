#!/usr/bin/env python
# coding: utf-8

import collections
import csv
import io

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras import backend as K
from tensorflow_addons.layers import WeightNormalization


## Callbacks
class LearningRateLogger(CSVLogger):

    def __init__(self, filename):
        super(CSVLogger, self).__init__()
        self.filename = filename

    def on_train_begin(self, logs=None):
        self.csv_file = io.open(self.filename, mode="a")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=['learning_rate'])

    def on_epoch_begin(self, epoch, logs=None):
        self.lr = float(K.get_value(self.model.optimizer.lr))

    def on_epoch_end(self, epoch, logs=None):
        row_dict = collections.OrderedDict({'learning_rate': self.lr})
        self.writer.writerow(row_dict)
        self.csv_file.flush()


class SepDenseLayer(tf.keras.Model):
    def __init__(self, num_input_dim, window_size, output_size, kernel_initializer, activation, use_bias, name):
        super(SepDenseLayer, self).__init__()
        self.num_input_dim = num_input_dim
        self.window_size = window_size
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.use_bias = use_bias
        self.t_name = name

        self.activation_layer = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        self.w = self.add_weight(name=F"sep_dense_{self.t_name}_weights",
                                 shape=(self.num_input_dim, self.window_size, self.output_size),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        if self.use_bias:
            self.b = self.add_weight(name=F"sep_dense_{self.t_name}_bias",
                                     shape=(self.num_input_dim, self.output_size),
                                     initializer='zeros',
                                     trainable=True)

    def call(self, input):
        x = tf.matmul(input, self.w)
        x = tf.squeeze(x, -2)
        if self.use_bias:
            x = tf.math.add(x, self.b)
        x = self.activation_layer(x)
        return x


class DownsampleLayerWithAttention(tf.keras.Model):
    def __init__(self, num_output_time_series, window_size, kernel_size, output_size, kernel_initializer, activation):
        super(DownsampleLayerWithAttention, self).__init__()
        self.num_output_time_series = num_output_time_series
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer

        self.down_tcn = tf.keras.layers.Conv1D(filters=num_output_time_series, kernel_size=kernel_size,
                                               padding="causal",
                                               kernel_initializer=kernel_initializer, name=F"downsample_tcn")
        self.weight_norm_down_tcn = WeightNormalization(self.down_tcn, data_init=False, name=F"wn_downsample_tcn")

        self.query_dense_layer = tf.keras.layers.Dense(output_size)

        self.key_dense_layer = SepDenseLayer(num_output_time_series, window_size, window_size, kernel_initializer,
                                             activation=activation, use_bias=True, name="key")

        self.value_dense_layer = SepDenseLayer(num_output_time_series, window_size, window_size, kernel_initializer,
                                               activation=activation, use_bias=False, name="value")
        self.post_attention_layer = DotAttentionLayer(window_size)

    def call(self, input_tensors):
        tcn_out = self.weight_norm_down_tcn(input_tensors[0])
        tcn_out = tf.transpose(tcn_out, [0, 2, 1])

        or_input = tf.transpose(input_tensors[1], [0, 2, 1])

        query = self.query_dense_layer(tcn_out)
        key = self.key_dense_layer(tf.expand_dims(or_input, -2))
        value = self.value_dense_layer(tf.expand_dims(or_input, -2))

        x, distribution = self.post_attention_layer([query, value, key])
        return x, distribution


class DotAttentionLayer(tf.keras.Model):
    def __init__(self, scale_value):
        super(DotAttentionLayer, self).__init__()
        self.scale_value = tf.cast(scale_value, tf.float32)

    def call(self, tensors):
        value = tf.expand_dims(tensors[1], -1)
        key = tf.expand_dims(tensors[2], -1)
        query = tf.expand_dims(tensors[0], -1)
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.sqrt(self.scale_value)
        distribution = tf.nn.softmax(scores)
        output = tf.squeeze(tf.matmul(distribution, value), -1)
        return output, distribution


class BasicTCNBlock(tf.keras.Model):
    def __init__(self, block_num, filter_num, kernel_size, dilation_rate, window_size, use_bias, kernel_initializer,
                 dropout_rate,
                 dropout_format, activation, final_activation):
        super(BasicTCNBlock, self).__init__()

        self.dropout_rate = dropout_rate
        valid_dropout_formats = {"channel", "timestep", "all"}
        if dropout_format not in valid_dropout_formats:
            raise ValueError("Dropout format must be one of %r." % valid_dropout_formats)
        if dropout_format == "channel":
            self.noise_shape = [1, filter_num]
        elif dropout_format == "timestep":
            self.noise_shape = [window_size, 1]
        else:
            self.noise_shape = [window_size, filter_num]

        self.tcn_1 = tf.keras.layers.Conv1D(filters=filter_num, kernel_size=kernel_size, padding="causal",
                                            dilation_rate=dilation_rate, use_bias=use_bias,
                                            kernel_initializer=kernel_initializer, name=F"{block_num}_tcn_1")
        self.weight_norm_layer_1 = WeightNormalization(self.tcn_1, data_init=False, name=F"{block_num}_wn_1")

        self.tcn_2 = tf.keras.layers.Conv1D(filters=filter_num, kernel_size=kernel_size, padding="causal",
                                            dilation_rate=dilation_rate, use_bias=use_bias,
                                            kernel_initializer=kernel_initializer, name=F"{block_num}_tcn_2")
        self.weight_norm_layer_2 = WeightNormalization(self.tcn_2, data_init=False, name=F"{block_num}_wn_2")

        self.tcn_3 = tf.keras.layers.Conv1D(filters=filter_num, kernel_size=1, padding="causal",
                                            dilation_rate=dilation_rate, use_bias=use_bias,
                                            kernel_initializer=kernel_initializer, name=F"{block_num}_tcn_3")
        self.weight_norm_layer_3 = WeightNormalization(self.tcn_3, data_init=False, name=F"{block_num}_wn_3")

        self.dropout_layer_1 = tf.keras.layers.Dropout(rate=self.dropout_rate, noise_shape=self.noise_shape,
                                                       name=F"{block_num}_dropout_1")

        self.dropout_layer_2 = tf.keras.layers.Dropout(rate=self.dropout_rate, noise_shape=self.noise_shape,
                                                       name=F"{block_num}_dropout_2")

        self.activation = tf.keras.layers.Activation(activation)

        self.final_activation = tf.keras.layers.Activation(final_activation)

    def call(self, input_tensor):

        x = self.weight_norm_layer_1(input_tensor)
        x = self.activation(x)
        x = self.dropout_layer_1(x)
        x = self.weight_norm_layer_2(x)
        x = self.activation(x)
        x = self.dropout_layer_2(x)
        res = self.weight_norm_layer_3(input_tensor)
        x = tf.math.add(res, x)
        x = self.final_activation(x)
        return x


class TCNStack(tf.keras.Model):
    def __init__(self, layer_num, filter_num, kernel_size, window_size,
                 use_bias,
                 kernel_initializer, dropout_rate, dropout_format, activation, final_activation,
                 final_stack_activation):
        super(TCNStack, self).__init__()
        self.kernel_size = kernel_size
        self.filter_num = filter_num
        self.use_bias = use_bias
        self.window_size = window_size
        self.layer_num = layer_num
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.dropout_format = dropout_format
        self.activation = activation
        self.final_activation = final_activation
        self.final_stack_activation = final_stack_activation

        # Create stack of TCN layers
        self.block_seq = tf.keras.models.Sequential()

    def build(self, input_shape):
        for i in range(self.layer_num - 1):
            self.block_seq.add(
                BasicTCNBlock(i, self.filter_num, self.kernel_size, 2 ** i, self.window_size,
                              self.use_bias, self.kernel_initializer, self.dropout_rate, self.dropout_format,
                              self.activation, self.final_activation))
        self.block_seq.add(
            BasicTCNBlock(self.layer_num - 1, self.filter_num, self.kernel_size, 2 ** (self.layer_num - 1),
                          self.window_size,
                          self.use_bias, self.kernel_initializer, self.dropout_rate, self.dropout_format,
                          self.activation, self.final_stack_activation))

    def call(self, input_tensor):
        x = self.block_seq(input_tensor)
        return x
