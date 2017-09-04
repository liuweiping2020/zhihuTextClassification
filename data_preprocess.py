# coding:utf-8
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
np.random.seed(1024)
stdo = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdo

from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,MultiLabelBinarizer

CHAR_EMBEDDING_FILE = './ieee_zhihu_cup/char_embedding.txt'
WORD_EMBEDDING_FILE = './ieee_zhihu_cup/word_embedding.txt'
TRAIN_DATA_FILE = './ieee_zhihu_cup/question_train_set.txt'
TEST_DATA_FILE = './ieee_zhihu_cup/question_eval_set.txt'
TOPIC_INFO_FILE = './ieee_zhihu_cup/topic_info.txt'
TRAIN_LABEL_FILE = './ieee_zhihu_cup/question_topic_train_set.txt'
MAX_SEQUENCE_LENGTH = 40


MAX_NB_CHARS = 10000             
MAX_NB_WORDS = 400000
EMBEDDING_DIM = 256


# Char_embedding
texts = []
with open(CHAR_EMBEDDING_FILE,'r') as f:
    cnt = 0
    for line in f.readlines():
        texts.append(line.split())
        cnt += 1
        if cnt % 1000 == 0:
            print cnt
texts.pop(0)     
embedding_matrix = pd.DataFrame(texts)
print embedding_matrix.shape
char_index = pd.DataFrame(embedding_matrix[0])
char_index.to_pickle('./cache/char_index.pkl')
del embedding_matrix[0]
embedding_matrix = embedding_matrix.as_matrix()
char_embedding_matrix_path = './cache/char_embedding_matrix.npy'
np.save(char_embedding_matrix_path, embedding_matrix)


# Word embedding
texts = []
with open(WORD_EMBEDDING_FILE,'r') as f:
    cnt = 0
    for line in f.readlines():
        texts.append(line.split())
        cnt += 1
        if cnt % 1000 == 0:
            print cnt
texts.pop(0)
embedding_matrix = pd.DataFrame(texts)
print embedding_matrix.shape
word_index = pd.DataFrame(embedding_matrix[0])
word_index.to_pickle('./cache/word_index.pkl')
del embedding_matrix[0]
embedding_matrix = embedding_matrix.as_matrix()
word_embedding_matrix_path = './cache/word_embedding_matrix.npy'
np.save(word_embedding_matrix_path, embedding_matrix)
del texts,embedding_matrix

# question_train_set
reader = pd.read_table('./ieee_zhihu_cup/question_train_set.txt',sep='\t',header=None)
print(reader.iloc[0:2])
# question_topic_eval_set
topic_reader = pd.read_table('./ieee_zhihu_cup/question_topic_train_set.txt',sep='\t',header=None)
print(topic_reader.iloc[0:2])
data_topic = pd.concat([reader.ix[:,2],reader.ix[:,4], topic_reader.ix[:,1]], axis=1, ignore_index=True)
print(data_topic.iloc[0:3])
# topic_info
label_reader = pd.read_table('./ieee_zhihu_cup/topic_info.txt',sep='\t',header=None)
print(label_reader.iloc[0:3])

# Build a topic_dict for label2topic
labels = list(label_reader.iloc[:,0])
my_labels = []
for label in labels:
    my_labels.append(label)    
topic_dict = {}
for i,label in enumerate(my_labels):
    topic_dict[label] = i
    
for i in tqdm(xrange(data_topic.shape[0])):
    new_label = ''
    temp_topic = data_topic.iloc[i][2].split(',')
    for topic in temp_topic:
        label_num = topic_dict[int(topic)]
        new_label = new_label + str(label_num) + ','
    data_topic.iloc[i][2] = new_label[:-1]
print(data_topic.iloc[:3])

# Save labels
label_path = './cache/train_label.csv'
train_label = data_topic.iloc[:,2]
train_label.to_csv(label_path,header=None)
print("Train label save in "+label_path)

# Process title and description
def get_word_id(data, max_document_length=80):
    try: 
        word_dict
    except:
        word_dict = pd.read_pickle('./cache/word_dict.pkl')
    text = []
    for line in data:
        try:
            text.append(line.split(','))
        except:
            text.append(' ')
    x = []
    for line in tqdm(text):
        line_len = len(line)
        text2num = []
        for i in xrange(max_document_length):
            if(i < line_len):
                try:
                    text2num.append(word_dict[line[i]]) 
                except:
                    text2num.append(0) 
            else:
                text2num.append(0) 
        x.append(text2num)
    x = np.array(x)
    return x

title_path = './cache/train_data_title.npy'
desc_path = './cache/train_data_desc.npy'
max_title_length = 50
max_desc_length = 150

# Save title data
x_title = get_word_id(data_topic.iloc[:,0], max_document_length=max_title_length)
print("x_title_shape:",x_title.shape)
np.save(title_path, x_title)
print('title data saved.')
# Save desc data
x_desc = get_word_id(data_topic.iloc[:,1], max_document_length=max_desc_length)
print("x_desc_shape:",x_desc.shape)
np.save(desc_path, x_desc)
print('desc data saved.')

# Split train data and validation data
title_path = './cache/train_data_title.npy'
desc_path = './cache/train_data_desc.npy'
label_path = './cache/train_label.csv'

print("Loading title  data ...")
title_data = np.load(title_path)
print("Loading description  data ...")
desc_data = np.load(desc_path)
print ("Loading label...")
y = pd.read_csv(label_path,header=None)
y = np.array(y.iloc[:,1])

# Shuffle
data_size = y.shape[0]
shuffle_indices = np.random.permutation(np.arange(data_size))
title_data = title_data[shuffle_indices]
desc_data = desc_data[shuffle_indices]
y = y[shuffle_indices]

dev_ratio = 0.1
validation_size = int(-1 * dev_ratio*data_size)

train_title_data = title_data[:validation_size]
valid_title_data = title_data[validation_size:]
train_desc_data = desc_data[:validation_size]
valid_desc_data = desc_data[validation_size:]
y_train = y[:validation_size]
y_valid = y[validation_size:]

# Save
np.save('./cache/train/new_train_title.npy',train_title_data)
np.save('./cache/valid/new_valid_title.npy',valid_title_data)
print"Title data saved."

np.save('./cache/train/new_train_desc.npy', train_desc_data)
np.save('./cache/valid/new_valid_desc.npy', valid_desc_data)
print"Desc data saved."

np.save('./cache/train/new_y_train.npy',y_train)
np.save('./cache/valid/new_y_valid.npy',y_valid)
print"Labels saved."