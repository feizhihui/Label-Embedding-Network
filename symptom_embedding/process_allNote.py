# encoding=utf-8
import csv
import pickle

with open('../PKL/symptom_dict.pkl', 'rb') as file:
    symptom_dict = pickle.load(file)

hadm2symptom = dict()
with open('../DATA/allNotes.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for i, columns in enumerate(csv_reader):
        if i == 0: continue
        hadm = columns[1]
        symptom_list = columns[-1].lower().strip('|').split('|')
        symptom_ids = []
        for symptom in symptom_list:
            symptom_key = symptom.split('#')[0]
            if symptom_key in symptom_dict:
                symptom_ids.append(symptom_dict[symptom_key])
        print(symptom_ids)
        hadm2symptom[hadm] = symptom_ids

with open('../PKL/hadm2symptom.pkl', 'wb') as file:
    pickle.dump(hadm2symptom, file)
