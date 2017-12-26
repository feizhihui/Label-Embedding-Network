# encoding=utf-8
import tensorflow as tf
import data_input
from text_cnn import TextCNN
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

Reader = data_input.data_master()

learning_rate = 0.001
batch_size = 256
epoch_num_cnn = 90
keep_pro = 0.9

model = TextCNN(Reader.embeddings)


def validataion():
    # model.prediction_fused
    print('begin to test:')
    outputs = []
    for i in range(0, Reader.test_size, batch_size):
        test_X_batch = Reader.test_X[i:i + batch_size]
        output = sess.run(model.prediction_cnn,
                          feed_dict={model.input_x: test_X_batch, model.label_x: Reader.label_x,
                                     model.dropout_keep_prob: 1.0})
        outputs.append(output)

    outputs = np.concatenate(outputs, axis=0)

    MiP, MiR, MiF, P_NUM, T_NUM = micro_score(outputs, Reader.test_Y)
    print(">>>>>> Final Result:  PredictNum:%.2f, TrueNum:%.2f" % (P_NUM, T_NUM))
    print(">>>>>> Micro-Precision:%.3f, Micro-Recall:%.3f, Micro-F Measure:%.3f" % (MiP, MiR, MiF))
    # MaP, MaR, MaF = macro_score(outputs, test_labels)
    # print(">>>>>> Macro-Precision:%.3f, Macro-Recall:%.3f, Macro-F Measure:%.3f" % (MaP, MaR, MaF))


def micro_score(output, label):
    N = len(output)
    total_P = np.sum(output)
    total_R = np.sum(label)
    TP = float(np.sum(output * label))
    MiP = TP / max(total_P, 1e-6)
    MiR = TP / max(total_R, 1e-6)
    MiF = 2 * MiP * MiR / (MiP + MiR)
    return MiP, MiR, MiF, total_P / N, total_R / N


def macro_score(output, label):
    total_P = np.sum(output, axis=0)
    total_R = np.sum(label, axis=0)
    TP = np.sum(output * label, axis=0)
    MiP = np.mean(TP / np.maximum(total_P, 1e-12))
    MiR = np.mean(TP / np.maximum(total_R, 1e-12))
    MiF = 2 * MiP * MiR / (MiP + MiR)
    return MiP, MiR, MiF


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('pretraining CNN Part')
    for epoch in range(epoch_num_cnn):
        Reader.shuffle()
        lr = learning_rate * (0.98 ** epoch)
        for iter, idx in enumerate(range(0, Reader.train_size, batch_size)):
            batch_X = Reader.train_X[idx:idx + batch_size]
            batch_Y = Reader.train_Y[idx:idx + batch_size]
            loss, output, _ = sess.run([model.loss_cnn, model.prediction_cnn, model.optimizer_cnn],
                                       feed_dict={model.input_x: batch_X, model.y: batch_Y,
                                                  model.label_x: Reader.label_x,
                                                  model.dropout_keep_prob: keep_pro, model.lr: lr})
            if iter % 100 == 0:
                print("===CNNPart===")
                MiP, MiR, MiF, P_NUM, T_NUM = micro_score(output, batch_Y)
                print("epoch:%d  iter:%d, mean loss:%.3f,  PNum:%.2f, TNum:%.2f" % (
                    epoch + 1, iter + 1, loss, P_NUM, T_NUM))
                print("Micro-Precision:%.3f, Micro-Recall:%.3f, Micro-F Measure:%.3f" % (MiP, MiR, MiF))
        if epoch >= 15:
            validataion()
