# encoding=utf-8


import csv
import re
import pickle
import numpy as np

embedding_size = 128

with open('../PKL/lookup_dict.pkl', 'rb') as file:
    lookup_dict = pickle.load(file)

lookup_matrix = np.load('../PKL/lookup_matrix.npy')

symptom_dict = dict()  # C0011991 ~ 789
symptom_matrix = []
with open('../DATA/symptomList.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for i, columns in enumerate(csv_reader):
        if i == 0: continue
        symptom_key = columns[1].lower()
        symptom_str = columns[-1].lower()
        word_list = re.findall(r'[a-z]+', symptom_str)
        print(symptom_key, symptom_str, word_list)  #
        symptom_dict[symptom_key] = i - 1

        symptom_vector = np.zeros([embedding_size])
        count = 0
        for word in word_list:
            if word in lookup_dict:
                count += 1
                symptom_vector += lookup_matrix[lookup_dict[word]]
        symptom_vector = symptom_vector / count
        symptom_matrix.append(symptom_vector)

np.save('../PKL/symptom_matrix.npy', np.array(symptom_matrix, dtype=np.float32))
with open('../PKL/symptom_dict.pkl', 'wb') as file:
    pickle.dump(symptom_dict, file)
