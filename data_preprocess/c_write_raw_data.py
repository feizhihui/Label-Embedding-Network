# encoding=utf-8
import pickle
import pandas as pd
import time
import re

term_pattern = re.compile('[A-Za-z]+')

with open('../PKL/hadm2codes.pkl', 'rb') as file:
    hadm2codes = pickle.load(file)

stop_word_set = set()
with open('../DATA/stopwords.txt', 'r') as file:
    for line in file.readlines():
        stop_word_set.add(line.strip())

print('loading reorganize raw_data module')
start_time = time.time()

noteeventFile = '../DATA/NOTEEVENTS.csv'

table = pd.read_csv(noteeventFile, usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CATEGORY', 'DESCRIPTION', 'TEXT'],
                    na_filter=False,
                    dtype={'SUBJECT_ID': str, 'HADM_ID': str, 'CHARTDATE': str, 'CATEGORY': str, 'DESCRIPTION': str,
                           'TEXT': str})

print('raw table lens:', len(table))
# drop the note without HADM_ID info
table = table[table['HADM_ID'] != '']
table = table[(table['CATEGORY'] == 'Discharge summary') & (table['DESCRIPTION'] == 'Report')]

print('generated discharge summary:', len(table))

print(time.time() - start_time)


# preprocess the note,delete some format string
def filter_note(raw_dsum):
    raw_dsum = re.sub(r'\n', ' ', raw_dsum)
    raw_dsum = re.sub(r'\s+', ' ', raw_dsum)
    raw_dsum = re.sub(r'\[[^\]]+\]', ' ', raw_dsum)
    raw_dsum = re.sub(r'admission date:', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'discharge date:', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'date of birth:', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'sex:', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'service:', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'dictated by:.*$', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'completed by:.*$', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'signed electronically by:.*$', ' ', raw_dsum, flags=re.I)
    tokens = [token.lower() for token in re.findall(term_pattern, raw_dsum) if
              token.lower() not in stop_word_set and len(token) > 2]
    return ' '.join(tokens)


with open('../DATA/MIMIC3_RAW_DSUMS', 'w') as file:
    file.write('"subject_id"|"hadm_id"|"charttime"|"category"|"title"|"icd9_codes"|"text"\n')
    for line in table.values:
        # line is a ndarray
        subject_id = line[0]
        hadm_id = line[1]
        chartdate = line[2]
        categoty = line[3]
        note_text = str.lower(line[5]).strip()
        if hadm_id not in hadm2codes:
            print("NOTEEVENTS.csv conclues", hadm_id, "but can't find in DIAGNOSE_ICD.csv")
            continue
        flist = []
        for code in hadm2codes[hadm_id]:
            flist.append(code)

        file.write(subject_id + '|')
        file.write(hadm_id + '|"')
        file.write(chartdate + '"|"')
        file.write(categoty + '"|"')
        file.write('"|"')
        # write codes
        file.write(flist[0])
        for fcode in flist[1:]:
            file.write(',' + fcode)
        file.write('"|"')
        note_text = filter_note(note_text)
        file.write(note_text + '\n')

print(time.time() - start_time)
