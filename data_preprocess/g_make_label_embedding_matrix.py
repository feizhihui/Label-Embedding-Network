# encoding=utf-8
import pickle
import numpy as np

with open('../PKL/code_dict.pkl', 'rb') as file:
    code_dict = pickle.load(file)

with open('../PKL/label2des.pkl', 'rb') as file:
    label2des = pickle.load(file)

with open('../PKL/lookup_dict.pkl', 'rb') as file:
    lookup_dict = pickle.load(file)


def filter_desc(desc):
    return [word for word in desc if word in lookup_dict]


reverse_dict = dict(zip(code_dict.values(), code_dict.keys()))

label_embedding_matrix = []
for i in range(len(reverse_dict)):
    code = reverse_dict[i]
    print(i, code)
    if code in label2des:
        print('find code', code)
        description = label2des[code]
    else:
        description = []
        print('find code', end=',')
        for j in range(10):
            code_5 = code + str(j)
            if code_5 in label2des:
                print(code_5, end=',')
                description.extend(label2des[code_5])
        print()
    label_embedding_matrix.append(filter_desc(description))

label_embedding_matrix = np.array(label_embedding_matrix)

print(label_embedding_matrix)
print(label_embedding_matrix.shape)

with open('../PKL/label_embedding_matrix.pkl', 'wb') as file:
    pickle.dump(label_embedding_matrix, file)
