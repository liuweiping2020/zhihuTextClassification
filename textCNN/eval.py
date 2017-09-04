# coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from six.moves import xrange

tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 256)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 256)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")

tf.flags.DEFINE_bool('is_train',False,"True for training, False for inference")
tf.flags.DEFINE_bool('loss_weighted',False,"True for weighting the loss")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

x_title_dim = 50
x_desc_dim = 120
y_dim = 1999

input_x_title = tf.placeholder(tf.int32, [None, x_title_dim], name="input_x_title")
input_x_desc = tf.placeholder(tf.int32, [None, x_desc_dim], name="input_x_desc")
input_y = tf.placeholder(tf.float32, [None, y_dim], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
learning_rate = tf.placeholder(tf.float32, name="learning_rate")
pos_weight = tf.placeholder(tf.float32,[None, y_dim], name="pos_weight")

# 预处理时需要准备好npy格式的embedding矩阵
# 这里使用之前得分比较好的模型的Embedding矩阵作为初始化
word_embedding_matrix = np.load('./cache/trained_embedding_matrix.npy')

title_sequence_length = x_title_dim
desc_sequence_length = x_desc_dim

num_classes = y_dim
embedding_size = FLAGS.embedding_dim
filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))

num_filters = FLAGS.num_filters
node_fc_layer = 1024

is_train = FLAGS.is_train

# Embedding矩阵
Weights = tf.Variable(word_embedding_matrix, trainable=True, name="Weights")
tf.summary.histogram('Embedding layer', Weights)

# shape:[None, sequence_length, embedding_size]
embedded_chars_title = tf.nn.embedding_lookup(Weights, input_x_title)
embedded_chars_title_expanded = tf.expand_dims(embedded_chars_title, -1)

pooled_title_outputs = []
with tf.name_scope("title"):
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W_1 = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name="W_1")
            tf.summary.histogram("W_1", W_1)

            b_1 = tf.Variable(
                tf.constant(0.1, shape=[num_filters]), name="b_1")
            tf.summary.histogram("b_1", b_1)

            conv = tf.nn.conv2d(
                embedded_chars_title_expanded,
                W_1,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            conv = tf.nn.bias_add(conv, b_1)

            bn = batch_norm_layer(conv, train_phase=is_train, scope_bn="title/conv-%s/conv_bn"%filter_size)

            # Apply nonlinearity
            relu = tf.nn.relu(bn, name="relu")
            shape = relu.shape.as_list()
            h = tf.reshape(relu, [-1, shape[1], shape[3], shape[2]])
            tf.summary.histogram("relu", h)

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, title_sequence_length - (filter_size-1), 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            tf.summary.histogram("pool", pooled)
            pooled_title_outputs.append(pooled)

# ______________________________________________________________________
# _______________II_:________Description________________________________

embedded_chars_desc = tf.nn.embedding_lookup(Weights, input_x_desc)
embedded_chars_desc_expanded = tf.expand_dims(embedded_chars_desc, -1)

pooled_desc_outputs = []
with tf.name_scope("description"):
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W_1 = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name="W_1")
            tf.summary.histogram("W_1", W_1)

            b_1 = tf.Variable(
                tf.constant(0.1, shape=[num_filters]), name="b_1")
            tf.summary.histogram("b_1", b_1)

            conv = tf.nn.conv2d(
                embedded_chars_desc_expanded,
                W_1,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv_1")
            conv = tf.nn.bias_add(conv, b_1)

            bn = batch_norm_layer(conv, train_phase=is_train, scope_bn="description/conv-%s/conv_bn"%filter_size)

            # Apply nonlinearity
            relu = tf.nn.relu(bn, name="relu")
            shape = relu.shape.as_list()
            h_desc = tf.reshape(relu, [-1, shape[1], shape[3], shape[2]])
            tf.summary.histogram("relu", h_desc)

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h_desc,
                ksize=[1, desc_sequence_length - (filter_size-1), 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            tf.summary.histogram("pool", pooled)

            pooled_desc_outputs.append(pooled)
# -----------------------------------------------------------------------------------

pooled_outputs = pooled_title_outputs + pooled_desc_outputs

# Combine all the pooled features
num_filters_total = 2 * num_filters * len(filter_sizes)

h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
tf.summary.histogram("Concated_Flatten", h_pool_flat)

with tf.name_scope("Drop_out"):
    h_pool_flat_dropout = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

with tf.name_scope("FC"):
    # FC layer 1
    fc_W_1 = tf.get_variable(
        "FC/W_1",
        shape=[num_filters_total, node_fc_layer],
        initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram('W_1', fc_W_1)
    fc_b_1 = tf.Variable(tf.constant(0.1, shape=[node_fc_layer]), name="b_1")
    tf.summary.histogram('b_1', fc_b_1)

    bn_1 = batch_norm_layer(tf.matmul(h_pool_flat_dropout, fc_W_1, name="fc_hidden") + fc_b_1, train_phase=is_train, scope_bn='bn_1')
    hidden = tf.nn.relu(bn_1)

    # FC layer 2
    fc_W_2 = tf.get_variable(
        "FC/W_2",
        shape=[node_fc_layer, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram('W_2', fc_W_2)

    fc_b_2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_2")
    tf.summary.histogram('b_2', fc_b_2)

with tf.name_scope("logits"):
    logits = tf.nn.xw_plus_b(hidden, fc_W_2, fc_b_2, name="logits")
    tf.summary.histogram('logits', logits)

with tf.name_scope("cross_entropy_loss"):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=input_y)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

with tf.name_scope("loss"):
    if FLAGS.loss_weighted:
        loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=input_y, pos_weight=pos_weight))
    else:
        loss = cross_entropy_loss
tf.summary.scalar('loss', loss)

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 建立一个placeholder仅为了在tensorboard里同步观察score
with tf.name_scope("score"):
    valid_score = tf.placeholder(tf.float32, name="score")
    tf.summary.scalar('score', valid_score)

# 同上, 记录每次验证时validation set的平均loss
with tf.name_scope("valid_loss"):
    valid_loss = tf.placeholder(tf.float32, name="valid_loss")
    tf.summary.scalar('valid_loss', valid_loss)


# 导入question_train_set
reader = pd.read_table('./ieee_zhihu_cup/question_eval_set.txt', sep='\t', header=None)
print(reader.iloc[0:2])

x_title_length = 50

# 计算一段文本中最大词汇数
x_title_text = reader.iloc[:,2]
x_desc_text = reader.iloc[:,2]

word_dict = pd.read_pickle('./cache/word_dict.pkl')

# 按','切分数据
title_text = []
for line in x_title_text:
    try:
        title_text.append(line.split(','))
    except:
        # 其中有一行数据为空
        title_text.append(' ')

desc_text = []
for line in x_desc_text:
    try:
        desc_text.append(line.split(','))
    except:
        # 其中有一行数据为空
        desc_text.append(' ')

# 把title数据集变成编号的形式
x_title = []
for line in tqdm(title_text):
    line_len = len(line)
    text2num = []
    for i in xrange(x_title_length):
        if(i < line_len):
            try:
                text2num.append(word_dict[line[i]]) # 把词转为数字
            except:
                text2num.append(0) # 没有对应的词
        else:
            text2num.append(0) # 填充0
    x_title.append(text2num)
x_title = np.array(x_title)
x_title[:2]

# 把desc数据集变成编号的形式
max_desc_length = 120

x_desc = []
for line in tqdm(desc_text):
    line_len = len(line)
    text2num = []
    for i in xrange(max_desc_length):
        if(i < line_len):
            try:
                text2num.append(word_dict[line[i]]) # 把词转为数字
            except:
                text2num.append(0) # 没有对应的词
        else:
            text2num.append(0) # 填充0
    x_desc.append(text2num)
x_desc = np.array(x_desc)
x_desc[:1]

# 选择模型
checkpoint_file = "./models/0809/model-60000"
    
# CPU设置
config = tf.ConfigProto(device_count={"CPU": 5}, # limit to num_cpu_core CPU usage  
                inter_op_parallelism_threads = 3,   
                intra_op_parallelism_threads = 4,  # 可以设大一点
                log_device_placement=True)
config.gpu_options.allow_growth = True
    
with tf.Session(config=config) as sess:
        predict_top_5 = tf.nn.top_k(logits, k=5)
        sess.run(tf.global_variables_initializer())
        i = 0
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

        # Generate batches
        batches = batch_iter(list(zip(x_title, x_desc)), 1000, 1)
        predict_result = []

        for batch in batches:
            x_title_batch, x_desc_batch = zip(*batch)
            i = i + 1
            predict_logits = sess.run(logits, feed_dict={input_x_title:x_title_batch,
                                                         input_x_desc:x_desc_batch,
                                                          dropout_keep_prob:1.0})
            for pred in predict_logits:
                predict_result.append(np.argsort(pred)[-1:-6:-1])

            if (i%10==0):
                print ("Evaluation:step",i)

        pd.to_pickle(predict_result, "prediction.pkl")
        print "Result saved in ./prediction.pkl"




