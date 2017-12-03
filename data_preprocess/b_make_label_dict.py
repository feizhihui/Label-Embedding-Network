# encoding=utf-8
import re
import pickle
import csv

term_pattern = re.compile('[A-Za-z]+')
label2des = dict()

stop_word_set = set()
with open('../DATA/stopwords.txt', 'r') as file:
    for line in file.readlines():
        stop_word_set.add(line.strip())

with open('../DATA/D_ICD_DIAGNOSES.csv', 'r') as file:
    for row_id, colums in enumerate(csv.reader(file)):
        if row_id == 0:
            continue
        code, description = colums[1], colums[3]
        description = re.sub(r'\(at present\)', ' ', description, flags=re.I)
        description = re.sub(r'\[any form\]', ' ', description, flags=re.I)
        description = re.sub(r'\[|\]|\(|\)', ' ', description, flags=re.I)
        tokens = [token.lower() for token in re.findall(term_pattern, description) if
                  token.lower() not in stop_word_set and len(token) > 2]
        label2des[code] = tokens

for label in label2des:
    print(label2des[label])

print(len(label2des))

with open('../PKL/label2des.pkl', 'wb') as file:
    pickle.dump(label2des, file)
