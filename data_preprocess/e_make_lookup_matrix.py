# encoding=utf-8
import pickle
import numpy as np

lookup_dict = dict()
lookup_matrix = []
lookup_dict['#PADDING#'] = 0
lookup_matrix.append([0] * 128)
with open('../DATA/embeddings.128', 'r') as file:
    for rowid, line in enumerate(file.readlines()[1:]):
        columns = line.split()
        word = columns[0]
        embedding_vector = columns[1:]
        lookup_dict[word] = rowid + 1
        lookup_matrix.append(embedding_vector)

with open('../PKL/lookup_dict.pkl', 'wb') as file:
    pickle.dump(lookup_dict, file)
# with open('../PKL/lookup_matrix.pkl', 'wb') as file:
#     pickle.dump(lookup_matrix, file)
lookup_matrix = np.array(lookup_matrix, dtype=np.float32)
np.save('../PKL/lookup_matrix.npy', lookup_matrix)

print(len(lookup_dict))
print(lookup_matrix.shape)
