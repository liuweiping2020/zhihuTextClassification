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
import tensorflow.contrib.slim as slim
from utils import *

tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 256)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 256)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate (default: 1e-3)")

tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 8, "Number of training epochs (default: 10)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 200)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_bool('is_train', True, "True for training, False for inference")
tf.flags.DEFINE_bool('loss_weighted', False, "True for weighting the loss")

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

word_embedding_matrix = np.load('./cache/trained_embedding_matrix.npy')

title_sequence_length = x_title_dim
desc_sequence_length = x_desc_dim

num_classes = y_dim
embedding_size = FLAGS.embedding_dim
filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))

num_filters = FLAGS.num_filters
node_fc_layer = 1024

is_train = FLAGS.is_train
        
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

config = tf.ConfigProto(device_count={"CPU": 3},
                inter_op_parallelism_threads=2,
                intra_op_parallelism_threads=2,
                log_device_placement=True)
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

date = "0802"
model_save_path = "models/{}/model".format(date)
summary_save_path = "./summary/{}/".format(date)

adam_learning_rate = FLAGS.learning_rate
decay = 0.66

with tf.Session(config=config) as sess:
    predict_top_5 = tf.nn.top_k(logits, k=5)
    label_top_5 = tf.nn.top_k(input_y, k=5) 
    sess.run(tf.global_variables_initializer())
    
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(summary_save_path, sess.graph)
    valid_writer = tf.summary.FileWriter(summary_save_path+"validation", sess.graph)
    
    x_title_train, x_desc_train, y_train = load_data(model='train', shuffle=True)
    x_title_dev, x_desc_dev, y_dev = load_data(model='valid', shuffle=False)

    x_title_train = x_title_train[:, :x_title_dim]
    x_title_dev   = x_title_dev[:, :x_title_dim]
    x_desc_train  = x_desc_train[:, :x_desc_dim]
    x_desc_dev    = x_desc_dev[:, :x_desc_dim]
    
    i = 0
    adam_learning_rate /= decay
    
    for epoch in range(FLAGS.num_epochs):
        adam_learning_rate *= decay

        print("Training Title Data shape: {},{}".format(x_title_train.shape, x_desc_train.shape))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        
        batches = batch_iter(
            list(zip(x_title_train, x_desc_train,  y_train)), FLAGS.batch_size, shuffle=True)

        for batch in batches:
            i += 1
            x_title_batch, x_desc_batch, label_batch = zip(*batch)
            y_batch, y_index_batch = one_hot_coding(label_batch)
            loss_pos_weight = get_weights(label_batch, alpha=0.35,beta=0.33)
            
            sess.run([optimizer],feed_dict={input_x_title:x_title_batch,
                                            input_x_desc:x_desc_batch,
                                            input_y:y_batch,
                                            learning_rate:adam_learning_rate,
                                            pos_weight:loss_pos_weight,
                                            dropout_keep_prob:FLAGS.dropout_keep_prob})

            if (i % FLAGS.evaluate_every == 0):
                train_logits, train_predict_5, train_label_5, train_loss = sess.run([logits, predict_top_5, label_top_5, loss],
                                                                                    feed_dict={input_x_title:x_title_batch,                                                                         input_x_desc:x_desc_batch,                                                                             input_y:y_batch,                                                                               learning_rate:adam_learning_rate,                                                                          pos_weight:loss_pos_weight,                                                                          dropout_keep_prob:FLAGS.dropout_keep_prob})
                # list, size:(64,5) 
                train_predict_results = []
                for logit in train_logits:
                    train_predict_results.append(list(np.argsort(logit)[-1:-6:-1]))
                    
                print "predict:", train_predict_results[:2]
                print "labels:", y_index_batch[:2]
                train_predict_label_and_marked_label_list = []
                for predict, label in zip(train_predict_results, y_index_batch):
                    train_predict_label_and_marked_label_list.append((list(predict), list(label)))
                train_score = eval(train_predict_label_and_marked_label_list)
                
                print("Step {}:\ttrain loss:{}\t score:{}".format(i, train_loss, train_score))
                
                result = sess.run(merged,feed_dict={input_x_title:x_title_batch,
                                                    input_x_desc:x_desc_batch, 
                                                    input_y:y_batch,
                                                    learning_rate:adam_learning_rate,
                                                    pos_weight:loss_pos_weight,
                                                    valid_score:train_score,
                                                    valid_loss:train_loss,
                                                    dropout_keep_prob:FLAGS.dropout_keep_prob})
                train_writer.add_summary(result, i)

                if (i % (25* FLAGS.evaluate_every) == 0):
                    valid_scores_list = []
                    valid_loss_list = []
                    print "Evaluation: at %d step:"%i

                    valid_batches = batch_iter(
                        list(zip(x_title_dev, x_desc_dev, y_dev)), FLAGS.batch_size, 1)
                    
                    valid_predict_label_and_marked_label_list = []
                    
                    for valid_batch in valid_batches:
                        x_title_valid, x_desc_valid, label_valid = zip(*valid_batch)
                        y_valid, y_index_valid = one_hot_coding(label_valid)
                        pos_weight_valid = get_weights(label_valid, alpha=0.35)
                        valid_logits, predict_5, label_5, _loss = sess.run([logits, predict_top_5, label_top_5, loss],
                                                                           feed_dict={input_x_title:x_title_valid,
                                                                                    input_x_desc:x_desc_valid,
                                                                                    input_y:y_valid,
                                                                                    learning_rate:adam_learning_rate,
                                                                                    pos_weight:pos_weight_valid,
                                                                                    dropout_keep_prob:1.0})

                        valid_predict_results = []
                        for logit in valid_logits:
                            valid_predict_results.append(list(np.argsort(logit)[-1:-6:-1]))
    
                        for predict,label in zip(valid_predict_results,y_index_valid):
                            valid_predict_label_and_marked_label_list.append((list(predict),list(label)))
                        
                        valid_loss_list.append(_loss)

                    avg_valid_loss = sum(valid_loss_list) / len(valid_loss_list)
                    score = eval(valid_predict_label_and_marked_label_list)

                    print "loss:", _loss
                    print "score:", score
                    result = sess.run(merged,
                                      feed_dict={input_x_title:x_title_valid,
                                                        input_x_desc:x_desc_valid,
                                                        input_y:y_valid, 
                                                        learning_rate:adam_learning_rate,
                                                        pos_weight:pos_weight_valid,
                                                        valid_score:score, 
                                                        valid_loss:avg_valid_loss, 
                                                        dropout_keep_prob:1.0}) 
                    valid_writer.add_summary(result, i)

            if i % FLAGS.checkpoint_every == 0:
                # Only save the models which have the score>0.39
                if train_score >= 0.39:
                    path = saver.save(sess, model_save_path, global_step=i)
                    print("Saved model checkpoint to {}".format(path))
