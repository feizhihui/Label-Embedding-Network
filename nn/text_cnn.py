# encoding=utf-8
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

embedding_size = 128

sequence_lens = 700
label_lens = 7

class_num = 6984
text_filter_num = 64
label_filter_num = 32

# fixed size 3
filter_sizes = [1, 3, 5]
threshold = 0.2


class TextCNN(object):
    def __init__(self, embeddings):
        weights1 = {
            'wc1': tf.Variable(tf.truncated_normal([filter_sizes[0], embedding_size, text_filter_num], stddev=0.1)),
            'wc2': tf.Variable(tf.truncated_normal([filter_sizes[1], embedding_size, text_filter_num], stddev=0.1)),
            'wc3': tf.Variable(tf.truncated_normal([filter_sizes[2], embedding_size, text_filter_num], stddev=0.1))
        }

        biases1 = {
            'bc1': tf.Variable(tf.truncated_normal([text_filter_num], stddev=0.1)),
            'bc2': tf.Variable(tf.truncated_normal([text_filter_num], stddev=0.1)),
            'bc3': tf.Variable(tf.truncated_normal([text_filter_num], stddev=0.1))
        }

        weights2 = {
            'wc1': tf.Variable(tf.truncated_normal([filter_sizes[0], embedding_size, label_filter_num], stddev=0.1)),
            'wc2': tf.Variable(tf.truncated_normal([filter_sizes[1], embedding_size, label_filter_num], stddev=0.1))
        }

        biases2 = {
            'bc1': tf.Variable(tf.truncated_normal([label_filter_num], stddev=0.1)),
            'bc2': tf.Variable(tf.truncated_normal([label_filter_num], stddev=0.1))
        }
        # define placehold


        self.input_x = tf.placeholder(tf.int32, [None, sequence_lens])
        self.label_x = tf.placeholder(tf.int32, [class_num, label_lens])

        self.y = tf.placeholder(tf.float32, [None, class_num])
        self.lr = tf.placeholder(tf.float32)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("Input_CNN_Part"):
            input_W = tf.Variable(embeddings, name="W", dtype=tf.float32)
            input_embeddings = tf.nn.embedding_lookup(input_W, self.input_x)
            input_convs = self.multi_text_conv(input_embeddings, weights1, biases1)
            print('after multiply convolutions: ', input_convs)
            input_convs = tf.reshape(input_convs, [-1, 3 * text_filter_num])
            input_convs = tf.nn.dropout(input_convs, self.dropout_keep_prob)
            print('input_convs:', input_convs)

        with tf.name_scope("Label_CNN_Part"):
            label_W = tf.Variable(embeddings, name="W", dtype=tf.float32)
            label_embeddings = tf.nn.embedding_lookup(label_W, self.label_x)
            label_convs = self.multi_label_conv(label_embeddings, weights2, biases2)
            print('after multiply convolutions: ', label_convs)
            label_convs = tf.reshape(label_convs, [-1, 2 * label_filter_num])
            label_convs = tf.nn.dropout(label_convs, self.dropout_keep_prob)
            print('label_convs:', label_convs)
        # tf.tile(C, [1,1,tf.shape(A)[2]])
        with tf.name_scope("Joint_Part"):
            # [None,192]=>[None,1,192]
            input_convs = tf.expand_dims(input_convs, 1)
            # [None,1,192]=>[None,6984,192]
            input_convs = tf.tile(input_convs, [1, class_num, 1])

            # [6984,48]=>[1,6984,48]
            label_convs = tf.expand_dims(label_convs, 0)
            # [1,6984,48]=>[None,6984,48]
            label_convs = tf.tile(label_convs, [tf.shape(input_convs)[0], 1, 1])
            # [None,6984,240]
            fused_tensor = tf.concat([input_convs, label_convs], axis=2)
            fused_tensor = tf.reshape(fused_tensor, [-1, 2 * label_filter_num + 3 * text_filter_num])

            output = layers.fully_connected(fused_tensor, 1,
                                            weights_initializer=tf.truncated_normal_initializer(
                                                stddev=np.sqrt(2. / (2 * label_filter_num + 3 * text_filter_num))),
                                            # He_Normalization
                                            biases_initializer=tf.zeros_initializer(),
                                            activation_fn=None)
            output = tf.reshape(output, [-1, class_num])
            self.loss_cnn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=output))
            self.optimizer_cnn = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_cnn)
            self.score_cnn = tf.nn.sigmoid(output)
            ones = tf.ones_like(self.score_cnn)
            zeros = tf.zeros_like(ones)
            self.prediction_cnn = tf.cast(tf.where(tf.greater(self.score_cnn, threshold), ones, zeros), tf.int32)

    def conv1d(sef, x, W, b, sample_lens):
        x = tf.reshape(x, shape=[-1, sample_lens, embedding_size])
        x = tf.nn.conv1d(x, W, 1, padding='SAME')
        x = tf.nn.bias_add(x, b)
        # shape=(n,time_steps,filter_num)
        h = tf.nn.relu(x)

        print('conv size:', h.get_shape().as_list())

        pooled = tf.reduce_max(h, axis=1)
        print('pooled size:', pooled.get_shape().as_list())
        return pooled

    def multi_text_conv(self, x, weights, biases):
        # Convolution Layer
        conv1 = self.conv1d(x, weights['wc1'], biases['bc1'], sequence_lens)
        conv2 = self.conv1d(x, weights['wc2'], biases['bc2'], sequence_lens)
        conv3 = self.conv1d(x, weights['wc3'], biases['bc3'], sequence_lens)
        #  n*time_steps*(3*filter_num)
        convs = tf.concat([conv1, conv2, conv3], 1)
        return convs

    def multi_label_conv(self, x, weights, biases):
        # Convolution Layer
        conv1 = self.conv1d(x, weights['wc1'], biases['bc1'], label_lens)
        conv2 = self.conv1d(x, weights['wc2'], biases['bc2'], label_lens)
        #  n*time_steps*(3*filter_num)
        convs = tf.concat([conv1, conv2], 1)
        return convs
