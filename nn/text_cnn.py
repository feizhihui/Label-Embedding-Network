# encoding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import numpy as np

embedding_size = 128

sequence_lens = 700
label_lens = 10

class_num = 6984
symptom_num = 5915

text_filter_num = 64

hidden_size = 64

# fixed size 3
filter_sizes = [1, 3, 5]
threshold = 0.3  # 0.2


class TextCNN(object):
    def __init__(self, embeddings, sym_embedding):
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

        # define placehold

        self.input_x = tf.placeholder(tf.int32, [None, sequence_lens])
        self.input_s = tf.placeholder(tf.int32, [None, symptom_num])
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

        with tf.name_scope("Label_GRU_Part"):
            label_W = tf.Variable(embeddings, name="W", dtype=tf.float32, trainable=False)  # , trainable=False
            label_embeddings = tf.nn.embedding_lookup(label_W, self.label_x)
            label_encoder = self.BidirectionalGRUEncoder(label_embeddings, name='label_encoder')
            label_encoder = self.AttentionLayer(label_encoder, name='sent_attention')
            label_encoder = tf.nn.dropout(label_encoder, self.dropout_keep_prob)
            print('label_gru:', label_encoder)

        with tf.name_scope("Symptom_embedding_Part"):
            symptom_W = tf.Variable(sym_embedding, name="W", dtype=tf.float32, trainable=False)  # , trainable=False
            # symptom_embeddings = tf.nn.embedding_lookup(symptom_W, self.input_s)
            symptom_embeddings = tf.matmul(tf.cast(self.input_s, tf.float32), tf.cast(symptom_W, tf.float32))
            symptom_embeddings = tf.nn.l2_normalize(symptom_embeddings, 1)

            symptom_output = layers.fully_connected(symptom_embeddings, hidden_size, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.truncated_normal_initializer(
                                                        stddev=np.sqrt(2. / embedding_size)),
                                                    # He_Normalization
                                                    biases_initializer=tf.zeros_initializer())

        with tf.name_scope("Label_Attention"):
            # [n,192]=>[n,64]
            u = layers.fully_connected(input_convs, 2 * hidden_size, activation_fn=tf.nn.tanh,
                                       weights_initializer=tf.truncated_normal_initializer(
                                           stddev=np.sqrt(2. / (2 * hidden_size))),
                                       # He_Normalization
                                       biases_initializer=tf.zeros_initializer())
            u = tf.expand_dims(u, 1)  # [n,1,64]
            u = tf.tile(u, [1, class_num, 1])  # [n,6984,64]

            # [6984,64]=>[1,6984,64]
            h = tf.expand_dims(label_encoder, 0)
            h = tf.tile(h, [tf.shape(input_convs)[0], 1, 1])  # [n,6984,64]

            attn_weight = tf.reduce_sum(tf.multiply(u, h), axis=2, keep_dims=True)  # [n,6984,1]
            attn_output = tf.reshape(attn_weight, [-1, class_num])  # similarity
            alpha = tf.nn.softmax(attn_weight, dim=1)  # [n,6984,1]
            atten_label = tf.reduce_sum(tf.multiply(h, alpha), axis=1)  # [n,64]

        with tf.name_scope("Output_Part"):
            # fused_tensor = tf.concat([atten_label, input_convs, symptom_output], axis=1)
            fused_tensor = tf.concat([symptom_output], axis=1)
            output = layers.fully_connected(fused_tensor, class_num,
                                            weights_initializer=tf.truncated_normal_initializer(
                                                stddev=np.sqrt(2. / (3 * text_filter_num + 2 * hidden_size))),
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

    def BidirectionalGRUEncoder(self, inputs, name):
        # 输入inputs的shape是[batch_size*sent_in_doc, word_in_sent, embedding_size]
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(hidden_size)
            GRU_cell_bw = rnn.GRUCell(hidden_size)
            # fw_outputs和bw_outputs的size都是[batch_size*sent_in_doc, word_in_sent, embedding_size]
            #  tuple of (outputs, output_states)
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=self.length(inputs),
                                                                                 dtype=tf.float32)
            # outputs的size是[batch_size*sent_in_doc, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    # 输出的状态向量按权值相加
    def AttentionLayer(self, inputs, name):
        # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为2×hidden_szie
            # 一个context记录了所有的经过全连接后的word或者sentence的权重
            u_context = tf.Variable(tf.truncated_normal([hidden_size * 2]), name='u_context')
            # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, hidden_size * 2, activation_fn=tf.nn.tanh)
            # alpha shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output

    def length(self, sequences):
        # 动态展开
        used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
        seq_len = tf.reduce_sum(used, reduction_indices=1)
        self.seq_len = tf.cast(seq_len, tf.int32)
        return self.seq_len
