import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
import sys
sys.path.append("../data_preprocessing/")
from data import crop_impact_area, crop_crop, get_imp_nparr

def get_fp_arrays(x_train, y_train, inds):
    x = np.array([])
    y = np.array([])
    for i in range(inds.shape[0]):
        x = np.append(x, x_train[int(inds[i])])
        y = np.append(y, y_train[int(inds[i])])
        x = x.reshape(i+1, x_train.shape[1], x_train.shape[2], x_train.shape[3])
        y = y.reshape(i+1, y_train.shape[1], y_train.shape[2], y_train.shape[3])
    return x, y
    
def load_data(dtype='float32'):
    with tf.name_scope('data'):
        images = np.load('../data/images.npy')
        masks = np.load('../data/masks.npy')
        x_train = images[:1000]
        y_train = masks[:1000]
        x_train = x_train.astype(dtype)
        y_train = y_train.astype(dtype)
        x_val = images[1000:2000]
        y_val = masks[1000:2000]
        x_val = x_val.astype(dtype)
        y_val = y_val.astype(dtype)
        train = (x_train, y_train)
        val = (x_val, y_val)
        test = (images[2000:3000].astype(dtype), masks[2000:3000].astype(dtype))
        return train, val, test

def get_variable(weights, name, dtype=tf.float32):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    if dtype == tf.float16:
        init = tf.cast(init, tf.float32)
        var = tf.get_variable(name=name, initializer=init,  shape=weights.shape, dtype=tf.float32)
        return tf.cast(var, tf.float16)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape, dtype=tf.float32)
    return var

def weight_variable(shape, stddev=0.02, name=None, dtype=tf.float32):
    initial = tf.truncated_normal(shape, stddev=stddev, dtype=dtype)
    if name is None:
        if dtype == tf.float16:
            return tf.cast(tf.Variable(initial, dtype=dtype), tf.float16)
        return tf.Variable(initial, dtype=dtype)
    else:
        if dtype == tf.float16:
            return tf.cast(tf.get_variable(name, initializer=initial, dtype=dtype), tf.float16)
        return tf.get_variable(name, initializer=initial, dtype=dtype)
 
def bias_variable(shape, name=None, dtype=tf.float32):
    initial = tf.constant(0.0, shape=shape, dtype=dtype)
    if name is None:
        if dtype == tf.float16:
            return tf.cast(tf.Variable(initial, dtype=dtype), tf.float16)
        return tf.Variable(initial, dtype=dtype)
    else:
        if dtype == tf.float16:
            return tf.cast(tf.get_variable(name, initializer=initial, dtype=tf.float16), tf.float16)
        return tf.get_variable(name, initializer=initial, dtype=dtype)
 
def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)
 
def conv(inputs, kernel_size, num_outputs, name,
         stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu, dtype=tf.float32):
    
    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value
        kernel_shape   = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape   = [1, stride_size[0], stride_size[1], 1]
        
        weights = tf.get_variable('weights', kernel_shape, dtype, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], dtype, tf.constant_initializer(0.0))
        conv    = tf.nn.conv2d(inputs, weights, stride_shape, padding = padding)
        outputs = tf.nn.bias_add(conv, bias)
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
     return outputs
 
def conv2d_basic(x, W, bias, dtype=tf.float32):
    if dtype == tf.float16:
        x = tf.cast(x, tf.float16)
        W = tf.cast(W, tf.float16)
        bias = tf.cast(bias, tf.float16)
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    if dtype == tf.float16:
        return tf.cast(tf.nn.bias_add(conv, bias), tf.float16)
    return tf.nn.bias_add(conv, bias)
 
def conv2d_strided(x, W, b, dtype=tf.float32):
    if dtype == tf.float16:
        x = tf.cast(x, tf.float16)
        W = tf.cast(W, tf.float16)
        b = tf.cast(b, tf.float16)
     conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    if dtype == tf.float32:
        return tf.cast(tf.nn.bias_add(conv, b), tf.float16)
    return tf.nn.bias_add(conv, b)
 
def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2, dtype=tf.float32):
    # print x.get_shape()
    # print W.get_shape()
    batch_size = 32
    if dtype == tf.float16:
        x = tf.cast(x, tf.float16)
        W = tf.cast(W, tf.float16)
        b = tf.cast(b, tf.float16)
     if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    output_shape = tf.stack([output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    if dtype == tf.float16:
        return tf.cast(tf.nn.bias_add(conv, b), tf.float16)
    return tf.nn.bias_add(conv, b)
 
def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)
 
def max_pool_2x2(x, dtype=tf.float32):
    if dtype == tf.float16:
        return tf.cast(tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"), tf.float16)
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
 
def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
 
def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
 
def batch_norm(x, n_out, phase_train, scope='bn', is_train=True, decay=0.9, eps=1e-5, dtype=tf.float32):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=is_train, dtype=dtype)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02), trainable=is_train, dtype=dtype)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)
         def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
         mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    if dtype == tf.float16:
        return tf.cast(normed, tf.float16)
    return normed
 
def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))
 
def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))
 
def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)
