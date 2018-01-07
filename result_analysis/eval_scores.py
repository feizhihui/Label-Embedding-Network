# encoding=utf-8

import pickle
import numpy as np
import sklearn.metrics as metrics
import data_input


def validataion(scores):
    # model.prediction_fused
    outputs = infer_by_threshold(scores, threshold=0.2)
    print(outputs.shape)

    MiP, MiR, MiF, P_NUM, T_NUM = micro_score(outputs, Reader.test_Y)
    print(">>>>>> Final Result:  PredictNum:%.2f, TrueNum:%.2f" % (P_NUM, T_NUM))
    print(">>>>>> Micro-Precision:%.3f, Micro-Recall:%.3f, Micro-F Measure:%.3f" % (MiP, MiR, MiF))


def micro_score(output, label):
    N = len(output)
    total_P = np.sum(output)
    total_R = np.sum(label)
    TP = float(np.sum(output * label))
    MiP = TP / max(total_P, 1e-6)
    MiR = TP / max(total_R, 1e-6)
    MiF = 2 * MiP * MiR / max(MiP + MiR, 1e-6)
    return MiP, MiR, MiF, total_P / N, total_R / N


def infer_by_threshold(scores, threshold=0.5):
    threshold = threshold * np.ones([len(scores), 1])
    scores = (scores >= threshold).astype(np.int32)
    return scores


Reader = data_input.data_master()

with open('../PKL/scores.pkl', 'rb') as file:
    scores = pickle.load(file)
    scores = np.concatenate(scores, axis=0)
    validataion(scores)
