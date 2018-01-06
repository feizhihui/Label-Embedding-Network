# encoding=utf-8
import csv

with open('../DATA/allNotes.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for i, columns in enumerate(csv_reader):
        if i == 0: continue
        symptom_info = columns[-1].lower()
        print(columns[-1])
