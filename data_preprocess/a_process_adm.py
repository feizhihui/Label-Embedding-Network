# encoding=utf-8
import pickle

diagnosisFile = '../DATA/DIAGNOSES_ICD.csv'
hadm2codes = dict()
code_dict = dict()
with open(diagnosisFile, 'r') as file:
    file.readline()
    for line in file.readlines():
        tokens = line.strip().split(',')
        hadm_id = tokens[2]
        icd9code = tokens[4].strip('"')
        if icd9code == '':
            continue
        code_dict[icd9code] = code_dict.get(icd9code, 0) + 1
        if hadm_id not in hadm2codes:
            hadm2codes[hadm_id] = [icd9code]
        else:
            hadm2codes[hadm_id].append(icd9code)

print(len(hadm2codes))
print(len(code_dict))
with open('../PKL/hadm2codes.pkl', 'wb') as file:
    pickle.dump(hadm2codes, file)

import operator

code_sort = sorted(code_dict.items(), key=lambda x: x[1], reverse=True)
print(code_sort)
code_dict = dict()
for i, (code, num) in enumerate(code_sort):
    code_dict[code] = i
with open('../PKL/code_dict.pkl', 'wb') as file:
    pickle.dump(code_dict, file)
