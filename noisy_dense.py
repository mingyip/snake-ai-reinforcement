import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import functools
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras import activations
from keras import initializers
from keras import regularizers, constraints



def noisy_dense(inputs, units, bias_shape, b_i=None, activation=tf.nn.relu, noisy_distribution='factorised'):
    def f(e_list):
        return tf.multiply(tf.sign(e_list), tf.pow(tf.abs(e_list), 0.5))
    # dense1 = tf.layers.dense(tf.contrib.layers.flatten(relu5), activation=tf.nn.relu, units=50)
    if not isinstance(inputs, ops.Tensor):
        inputs = ops.convert_to_tensor(inputs, dtype='float')
        # dim_list = inputs.get_shape().as_list()
        # flatten_shape = dim_list[1] if len(dim_list) <= 2 else reduce(lambda x, y: x * y, dim_list[1:])
        # reshaped = tf.reshape(inputs, [dim_list[0], flatten_shape])
    if len(inputs.shape) > 2:
        inputs = tf.contrib.layers.flatten(inputs)

    w_i = tf.random_uniform_initializer(-0.1, 0.1)
    flatten_shape = 10*10*16#inputs.shape[1]
    weights = tf.get_variable('weights', shape=[flatten_shape, units], initializer=w_i)
    w_noise = tf.get_variable('w_noise', [flatten_shape, units], initializer=w_i)
    if noisy_distribution == 'independent':
        weights += tf.multiply(tf.random_normal(shape=w_noise.shape), w_noise)
    elif noisy_distribution == 'factorised':
        noise_1 = f(tf.random_normal(tf.TensorShape([flatten_shape, 1]), dtype=tf.float32))
        noise_2 = f(tf.random_normal(tf.TensorShape([1, units]), dtype=tf.float32))
        weights += tf.multiply(noise_1 * noise_2, w_noise)
    dense = tf.matmul(inputs, weights)
    if bias_shape is not None:
        assert bias_shape[0] == units
        biases = tf.get_variable('biases', shape=bias_shape, initializer=b_i)
        b_noise = tf.get_variable('b_noise', [1, units], initializer=b_i)
        if noisy_distribution == 'independent':
            biases += tf.multiply(tf.random_normal(shape=b_noise.shape), b_noise)
        elif noisy_distribution == 'factorised':
            biases += tf.multiply(noise_2, b_noise)
        return activation(dense + biases) if activation is not None else dense + biases
    return activation(dense) if activation is not None else dense




class NoiseDense(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NoiseDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernelSigma = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernelSigma',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.biasSigma = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='biasSigma',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
 
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        newW = K.add(self.kernel,self.kernelSigma)
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(NoiseDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
