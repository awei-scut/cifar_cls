#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


## 数据预处理
class CifarData():
    def __init__(self, data, labels):
        self._data = data
        self._labels = labels
        self._indicator = 0
        self._example_num = len(data)
        self._random_shuffle()
        self._deal_img()
        
    def _random_shuffle(self):
        p = np.random.permutation(self._example_num)
        self._data = self._data[p]
        self._labels = self._labels[p]
        
    def _deal_img(self):
        data = self._data.reshape(self._example_num, 3, 32, 32)
        data = data / 255
        self._data = np.transpose(data, [0, 2, 3, 1])
        new_label = np.zeros([self._example_num, 10])
        for i in range(self._example_num):
            new_label[i, self._labels[i]] = 1 
        self._labels = new_label
        
    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._example_num:
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        batch_data = self._data[self._indicator: end_indicator]
        batch_label = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator        
        return batch_data, batch_label

def get_default_parms():
    return tf.contrib.training.HParams(
        batch_size = 128,
        channels = [16, 32, 64, 128],
        learning_rate = 0.002,
    )

## conv封装
def conv2d_warpper(inputs, out_channel, name, training):
    def leaky_relu(x, alpha=0.1, name = ''):
        return tf.maximum(x, alpha * x, name=name)
    with tf.variable_scope(name):
        conv2d = tf.layers.conv2d(inputs, out_channel, [5, 5], strides=(2, 2), padding="SAME")
        bn = tf.layers.batch_normalization(conv2d, training=training)
        return leaky_relu(bn, name='output')

class NetWork():
    
    def __init__(self, hps):
        self._hps = hps
        self._reuse = False
        
    def build(self, training):
        
        self._input = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self._label = tf.placeholder(tf.int32, [None, 10])
        
        ## 搭建网络
        conv_input = self._input
        with tf.variable_scope('conv', reuse=self._reuse):
            for i in range(len(self._hps.channels)):
                conv_input = conv2d_warpper(conv_input, self._hps.channels[i], 'conv2d_%d' % i, training=training)
        fc_inputs = conv_input
        with tf.variable_scope('fc', reuse=self._reuse):
            flatten = tf.layers.flatten(fc_inputs)
            fc= tf.layers.dense(flatten, 512, name='fc')
            out = tf.layers.dense(fc, 10, name='output')
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=self._label))
        self._reuse = True
        return self._input, self._label, tf.nn.softmax(out), loss
    
    def build_op(self, loss):
        updata_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updata_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=self._hps.learning_rate).minimize(loss)
        return train_op
    
    def test_acc(self):
        input_ts , labels_ts, out, loss = self.build(True)
        correct_predict = tf.equal(tf.argmax(out, 1), tf.argmax(labels_ts, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        return input_ts, labels_ts, accuracy, loss
        

## 读取训练数据
cifar = unpickle('./cifar-10-batches-py/data_batch_1')
all_data = cifar[b'data']
all_label = np.array(cifar[b'labels'])
for i in range(1, 4):
    cifar = unpickle('./cifar-10-batches-py/data_batch_%d' % (i+1))
    all_data = np.concatenate((all_data, cifar[b'data']), axis=0)
    all_label = np.concatenate((all_label, np.array(cifar[b'labels'])), axis=0)
    cifar_data = CifarData(all_data, all_label)

## 读取测试数据
cifar2 = unpickle('./cifar-10-batches-py/data_batch_5')
test_data = cifar2[b'data']
test_label = np.array(cifar2[b'labels'])
cifar_test = CifarData(test_data, test_label)


hps = get_default_parms()
tf.reset_default_graph()
net = NetWork(hps)
input_ts, labels_ts, out, loss = net.build(True)
d, l, acc_test, loss_test = net.test_acc()
opt = net.build_op(loss)

## 保存模型
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.5 

saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, './checkpoints/myModel')
    step = 5000
    for i in range(1, step):
        batch_data, batch_label = cifar_data.next_batch(512)
        batch_test, batch_test_label = cifar_test.next_batch(512)
        ## 训练时
        sess.run(opt, feed_dict={input_ts:batch_data, labels_ts:batch_label})
        loss_value = sess.run(loss, feed_dict={input_ts:batch_data, labels_ts:batch_label})

        correct_predict = tf.equal(tf.argmax(out, 1), tf.argmax(labels_ts, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        train_acc = sess.run(accuracy, feed_dict={input_ts:batch_data, labels_ts:batch_label})
        
        ## 测试时
        
        test_acc = sess.run([acc_test, loss_test], feed_dict={d:batch_test, l:batch_test_label})
        
        print("step: %d" % i + " loss=" + str(loss_value) + " train_acc:" + str(train_acc)+ " test_acc: " + str(test_acc[0]) + " test_loss=" + str(test_acc[1]))
        
        ### 200step 保存一下模型
        if i % 200 == 0:
            saver.save(sess, "./checkpoints/myModel")
        


