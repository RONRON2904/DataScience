from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
import os
import re
from tensorflow.contrib.layers import xavier_initializer
import time
import utils
import tensorlayer as tl
tf.logging.set_verbosity(tf.logging.INFO)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
tf.Session(config=config)

tf.logging.set_verbosity(tf.logging.DEBUG)



IMAGE_SIZE = 128

Train, val, test = utils.load_data(dtype='float32')
X_train = Train[0]
y_train = Train[1]
X_val = val[0]
y_val = val[1]
X_test = test[0]
y_test = test[1]

t = tf.float32
tl.layers.LayersConfig.tf_dtype = t
sess = tf.InteractiveSession()
batch_size = 32
x = tf.placeholder(t, shape=[batch_size, 128, 128, 1])
y_ = tf.placeholder(t, shape=[batch_size, 128, 128, 1])

def u_net(image, phase_train, train=True, reuse=False, dtype=t):
    with tf.variable_scope("u_net", reuse=reuse):
        w1_1 = utils.weight_variable([3,3,int(image.shape[3]),32],name="w1_1", dtype=dtype)
        b1_1 = utils.bias_variable([32],name="b1_1", dtype=dtype)
        conv1_1 = utils.conv2d_basic(image,w1_1,b1_1, dtype=dtype)
        relu1_1 = tf.nn.relu(conv1_1, name="relu1_1")
        w1_2 = utils.weight_variable([3,3,32,32],name="w1_2", dtype=dtype)
        b1_2 = utils.bias_variable([32],name="b1_2", dtype=dtype)
        conv1_2 = utils.conv2d_basic(relu1_1,w1_2,b1_2, dtype=dtype)
        relu1_2 = tf.nn.relu(conv1_2, name="relu1_2")
        pool1 = utils.max_pool_2x2(relu1_2, dtype=dtype)
        bn1 = utils.batch_norm(pool1,pool1.get_shape()[3],phase_train,scope="bn1", is_train=train, dtype=dtype)

        w2_1 = utils.weight_variable([3,3,32,64],name="w2_1", dtype=dtype)
        b2_1 = utils.bias_variable([64],name="b2_1", dtype=dtype)
        conv2_1 = utils.conv2d_basic(bn1,w2_1,b2_1, dtype=dtype)
        relu2_1 = tf.nn.relu(conv2_1, name="relu2_1")
        w2_2 = utils.weight_variable([3,3,64,64],name="w2_2", dtype=dtype)
        b2_2 = utils.bias_variable([64],name="b2_2", dtype=dtype)
        conv2_2 = utils.conv2d_basic(relu2_1,w2_2,b2_2, dtype=dtype)
        relu2_2 = tf.nn.relu(conv2_2, name="relu2_2")
        pool2 = utils.max_pool_2x2(relu2_2, dtype=dtype)
        bn2 = utils.batch_norm(pool2,pool2.get_shape()[3],phase_train,scope="bn2", is_train=train, dtype=dtype)
        
        w3_1 = utils.weight_variable([3,3,64,128],name="w3_1", dtype=dtype)
        b3_1 = utils.bias_variable([128],name="b3_1", dtype=dtype)
        conv3_1 = utils.conv2d_basic(bn2,w3_1,b3_1, dtype=dtype)
        relu3_1 = tf.nn.relu(conv3_1, name="relu3_1")
        w3_2 = utils.weight_variable([3,3,128,128],name="w3_2", dtype=dtype)
        b3_2 = utils.bias_variable([128],name="b3_2", dtype=dtype)
        conv3_2 = utils.conv2d_basic(relu3_1,w3_2,b3_2, dtype=dtype)
        relu3_2 = tf.nn.relu(conv3_2, name="relu3_2")
        pool3 = utils.max_pool_2x2(relu3_2)
        bn3 = utils.batch_norm(pool3,pool3.get_shape()[3],phase_train,scope="bn3", is_train=train, dtype=dtype)

        w4_1 = utils.weight_variable([3,3,128,256],name="w4_1", dtype=dtype)
        b4_1 = utils.bias_variable([256],name="b4_1", dtype=dtype)
        conv4_1 = utils.conv2d_basic(bn3,w4_1,b4_1, dtype=dtype)
        relu4_1 = tf.nn.relu(conv4_1, name="relu4_1")
        w4_2 = utils.weight_variable([3,3,256,256],name="w4_2", dtype=dtype)
        b4_2 = utils.bias_variable([256],name="b4_2", dtype=dtype)
        conv4_2 = utils.conv2d_basic(relu4_1,w4_2,b4_2, dtype=dtype)
        relu4_2 = tf.nn.relu(conv4_2, name="relu4_2")
        bn4 = utils.batch_norm(relu4_2,relu4_2.get_shape()[3],phase_train,scope="bn4", is_train=train, dtype=dtype)
                
        W_t1 = utils.weight_variable([2, 2, 128, 256], name="W_t1", dtype=dtype)
        b_t1 = utils.bias_variable([128], name="b_t1", dtype=dtype)
        conv_t1 = utils.conv2d_transpose_strided(bn4, W_t1, b_t1, output_shape=tf.shape(relu3_2),dtype=dtype)
        merge1 = tf.concat([conv_t1,relu3_2],3)
        w5_1 = utils.weight_variable([3,3,256,128],name="w5_1", dtype=dtype)
        b5_1 = utils.bias_variable([128],name="b5_1", dtype=dtype)
        conv5_1 = utils.conv2d_basic(merge1,w5_1,b5_1, dtype=dtype)
        relu5_1 = tf.nn.relu(conv5_1, name="relu6_1")
        w5_2 = utils.weight_variable([3,3,128,128],name="w5_2", dtype=dtype)
        b5_2 = utils.bias_variable([128],name="b5_2", dtype=dtype)
        conv5_2 = utils.conv2d_basic(relu5_1,w5_2,b5_2,dtype=dtype)
        relu5_2 = tf.nn.relu(conv5_2, name="relu5_2")
        bn5 = utils.batch_norm(relu5_2,relu5_2.get_shape()[3],phase_train,scope="bn5", is_train=train, dtype=dtype)
                
        W_t2 = utils.weight_variable([2, 2, 64, 128], name="W_t2", dtype=dtype)
        b_t2 = utils.bias_variable([64], name="b_t2", dtype=dtype)
        conv_t2 = utils.conv2d_transpose_strided(bn5, W_t2, b_t2, output_shape=tf.shape(relu2_2),dtype=dtype)
        merge2 = tf.concat([conv_t2,relu2_2],3)
        w6_1= utils.weight_variable([3,3,128,64],name="w6_1", dtype=dtype)
        b6_1= utils.bias_variable([64],name="b6_1", dtype=dtype)
        conv6_1 = utils.conv2d_basic(merge2,w6_1,b6_1, dtype=dtype)
        relu6_1 = tf.nn.relu(conv6_1, name="relu6_1")
        w6_2 = utils.weight_variable([3,3,64,64],name="w6_2", dtype=dtype)
        b6_2 = utils.bias_variable([64],name="b6_2", dtype=dtype)
        conv6_2 = utils.conv2d_basic(relu6_1,w6_2,b6_2, dtype=dtype)
        relu6_2 = tf.nn.relu(conv6_2, name="relu6_2")
        bn6 = utils.batch_norm(relu6_2,relu6_2.get_shape()[3],phase_train,scope="bn6", is_train=train, dtype=dtype)
	
        W_t3 = utils.weight_variable([2, 2, 32, 64], name="W_t3", dtype=dtype)
        b_t3 = utils.bias_variable([32], name="b_t3", dtype=dtype)
        conv_t3 = utils.conv2d_transpose_strided(bn6, W_t3, b_t3, output_shape=tf.shape(relu1_2),dtype=dtype)
        merge3 = tf.concat([conv_t3,relu1_2],3)
        w7_1 = utils.weight_variable([3,3,64,32],name="w7_1", dtype=dtype)
        b7_1 = utils.bias_variable([32],name="b7_1", dtype=dtype)
        conv7_1 = utils.conv2d_basic(merge3,w7_1,b7_1, dtype=dtype)
        relu7_1 = tf.nn.relu(conv7_1, name="relu7_1")
        w7_2 = utils.weight_variable([3,3,32,32],name="w7_2", dtype=dtype)
        b7_2 = utils.bias_variable([32],name="b7_2", dtype=dtype)
        conv7_2 = utils.conv2d_basic(relu7_1,w7_2,b7_2, dtype=dtype)
        relu7_2 = tf.nn.relu(conv7_2, name="relu7_2")
        bn7 = utils.batch_norm(relu7_2,relu7_2.get_shape()[3],phase_train,scope="bn8", is_train=train, dtype=dtype)
                
        w8 = utils.weight_variable([1, 1, 32, 1], name="w8", dtype=dtype)
        b8 = utils.bias_variable([1],name="b8", dtype=dtype)
        conv8 = utils.conv2d_basic(bn7,w8,b8, dtype=dtype)
        return conv8

phase_test=tf.Variable(True,name="phase_train",trainable=False)
phase_train=tf.Variable(True,name="phase_train",trainable=False)
y = u_net(x, phase_train)
y2 = u_net(x, phase_test, train=False, reuse=tf.AUTO_REUSE)
if t == tf.float16:
    y_ = tf.cast(y_, tf.float32)
    y = tf.cast(y, tf.float32)
    y2 = tf.cast(y2, tf.float32)
cost = tl.cost.sigmoid_cross_entropy(y, y_, name='xentropy')
cost_test = tl.cost.sigmoid_cross_entropy(y2, y_, name='xentropy2')
train_params = tl.layers.get_variables_with_name('u_net', train_only=True, printable=False)

train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-4,
                                  use_locking=False).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess)
n_epoch = 100
print_freq = 1
for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, n_batch = 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
            err = sess.run(cost_test, feed_dict={x: X_train_a, y_: y_train_a})
            train_loss += err
            n_batch += 1
        print("   train loss: %f" % (train_loss / n_batch))
        val_loss, n_batch = 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            err = sess.run(cost_test, feed_dict={x: X_val_a, y_: y_val_a})
            val_loss += err
            n_batch += 1
        print("   val loss: %f" % (val_loss / n_batch))

print('Evaluation')
test_loss, n_batch = 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
    err = sess.run([cost_test], feed_dict={x: X_test_a, y_: y_test_a})
    test_loss += err
    n_batch += 1
print("   test loss: %f" % (test_loss / n_batch))



