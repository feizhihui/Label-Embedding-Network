# encoding=utf-8
import pickle
import numpy as np
import random

sequence_lens = 700
class_num = 6984
train_eval_split_line = 0.9
shuffle = False


class data_master:
    def __init__(self):
        self.embeddings = np.load('../PKL/lookup_matrix.npy')
        with open('../PKL/lookup_dict.pkl', 'rb') as file:
            self.word_dict = pickle.load(file)
        with open('../PKL/code_dict.pkl', 'rb') as file:
            self.code_dict = pickle.load(file)
        with open('../PKL/all_text.pkl', 'rb') as file:
            all_text = pickle.load(file)
        with open('../PKL/all_code.pkl', 'rb') as file:
            all_code = pickle.load(file)

        self.all_text = self.mapping_sequence(all_text)
        self.all_code = self.mapping_multi_hot(all_code)

        boundary = int(train_eval_split_line * len(all_text))
        self.train_X = self.all_text[:boundary]
        self.train_Y = self.all_code[:boundary]

        self.test_X = self.all_text[boundary:]
        self.test_Y = self.all_code[boundary:]

        self.train_size = boundary
        print('training size:', self.train_size)
        print('eval size:', len(all_text) - self.train_size)
        print(self.train_X.shape)

    def shuffle(self):
        permutation = list(range(self.train_size))
        random.shuffle(permutation)
        self.train_X = self.train_X[permutation]
        self.train_Y = self.train_Y[permutation]

    def mapping_sequence(self, batch_X):
        batch_X_matrix = np.zeros([len(batch_X), sequence_lens], dtype=np.int32)
        for i, sample in enumerate(batch_X):
            for j, word in enumerate(sample[:sequence_lens]):
                token = self.word_dict[word]
                batch_X_matrix[i, j] = token
        return batch_X_matrix

    def mapping_multi_hot(self, batch_Y):
        batch_Y_matrix = np.zeros([len(batch_Y), class_num], dtype=np.int32)
        for i, code_set in enumerate(batch_Y):
            for code in code_set:
                code = self.code_dict[code]
                batch_Y_matrix[i, code] = 1

        return batch_Y_matrix


if __name__ == '__main__':
    dataReader = data_master()
    print(dataReader.test_X)
    print(dataReader.test_Y)
    print(len(dataReader.test_Y))
    dataReader.shuffle()
    print(dataReader.mapping_sequence(dataReader.test_X))
    print(dataReader.mapping_multi_hot(dataReader.test_Y))
