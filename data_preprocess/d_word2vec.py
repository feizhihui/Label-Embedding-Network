# encoding=utf-8
import pickle

corpus = []
with open('../DATA/MIMIC3_RAW_DSUMS', 'r') as file:
    # file.write('"subject_id"|"hadm_id"|"charttime"|"category"|"title"|"icd9_codes"|"text"\n')
    for line in file.readlines()[1:]:
        text = line.split('|')[6]
        corpus.append(text.split())

print(len(corpus), corpus[-1])

with open('../PKL/label2des.pkl', 'rb') as file:
    label2des = pickle.load(file)

for label, des in label2des.items():
    corpus.append(des)

print(len(corpus), corpus[-1])
print('begin to train word2vec:')

from gensim.models import word2vec

# 引入日志配置
import logging
import time

start_time = time.time()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 构建模型
print('begin to train word2vec model...')
model = word2vec.Word2Vec(corpus, size=128, min_count=10)  # default size 100
model.save('../MODEL/word2vec.model')
# 保存词向量
model.wv.save_word2vec_format('../DATA/embeddings.128', binary=False)

end_time = time.time()
print((end_time - start_time) / 60, 'minutes')
