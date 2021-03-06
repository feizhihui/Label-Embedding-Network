# encoding=utf-8

import pickle
import numpy as np
import sklearn.metrics as metrics
import data_input


def validataion(scores):
    # model.prediction_fused
    print(scores.shape)
    # search best threshold
    thresholds = search_threshold(scores)
    thresholds = search_threshold(scores, thresholds)
    outputs = infer_by_threshold(scores, threshold=thresholds)
    # outputs = infer_by_threshold(scores, threshold=0.23)
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


def fast_micro_score(total_p, total_r, total_tp):
    N = len(total_p)
    total_P = np.sum(total_p)
    total_R = np.sum(total_r)
    TP = float(np.sum(total_tp))
    MiP = TP / max(total_P, 1e-6)
    MiR = TP / max(total_R, 1e-6)
    MiF = 2 * MiP * MiR / max(MiP + MiR, 1e-6)
    return MiP, MiR, MiF, total_P / N, total_R / N


def change_result(label, pred, threshold):
    pred = (pred >= threshold).astype(np.int32)
    p = np.sum(pred)
    r = np.sum(label)
    tp = np.sum(label * pred)
    return p, r, tp


def infer_by_threshold(scores, threshold=0.5):
    threshold = threshold * np.ones([scores.shape[1], 1])
    scores = (scores >= threshold.T).astype(np.int32)
    return scores


def search_threshold(scores, threshold=0.5):
    thresholds = threshold * np.ones([scores.shape[1], 1])
    assert len(thresholds) == 6984
    result = infer_by_threshold(scores, thresholds)
    total_p = np.sum(result, axis=0)
    total_r = np.sum(Reader.test_Y, axis=0)
    total_tp = np.sum(result * Reader.test_Y, axis=0)

    for i in range(scores.shape[1]):
        best_t = 0.99
        best_f_score = 0
        for t in range(0, 1000, 10):
            t = t / 1000.
            # f_score = metrics.f1_score((scores[:, i] >= t).astype(np.int32), Reader.test_Y[:, i])
            thresholds[i, 0] = t
            # result = infer_by_threshold(scores, thresholds)
            # achieve = micro_score(result, Reader.test_Y)

            total_p[i], total_r[i], total_tp[i] = change_result(Reader.test_Y[:, i], scores[:, i], t)
            achieve = fast_micro_score(total_p, total_r, total_tp)

            if best_f_score < achieve[2]:
                best_f_score = achieve[2]
                best_t = t
                p, r, tp = total_p[i], total_r[i], total_tp[i]
        total_p[i], total_r[i], total_tp[i] = p, r, tp
        thresholds[i, 0] = best_t
        if np.sum(Reader.test_Y[:, i]) == 0:
            auc = 0
        else:
            auc = metrics.roc_auc_score(Reader.test_Y[:, i], scores[:, i])
        print('code %d, threshold %.3f, best f-score %.4f, AUC %.4f' % (i, best_t, best_f_score, auc))
    return thresholds


Reader = data_input.data_master()

with open('../PKL/scores.pkl', 'rb') as file:
    scores = pickle.load(file)
    scores = np.concatenate(scores, axis=0)
    validataion(scores)
