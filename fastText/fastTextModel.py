# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
import math
from tqdm import tqdm
from six.moves import xrange
import tensorflow.contrib.slim as slim
from utils import *

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 256)")

tf.flags.DEFINE_float("drop_out_keep_prob", 0.5, "Keep prob")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate (default: 1e-3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 6, "Number of training epochs (default: 10)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 4000, "Save model after this many steps (default: 200)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_bool('is_train',True,"True for training, False for inference")
tf.flags.DEFINE_bool('is_weighted',False,"True for weighted loss, False for norm crossentropy")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


label_size = 1999
batch_size = FLAGS.batch_size
num_sampled = 10
decay_steps = 20000
decay_rate = 0.66
embedding_dim = FLAGS.embedding_dim
l2_lambda = FLAGS.l2_reg_lambda

x_title_dim = 40
x_desc_dim = 120
y_dim = 1999

hidden_unit = 1024

input_x_title = tf.placeholder(tf.int32, [None, x_title_dim], name="input_x_title")
input_x_desc  = tf.placeholder(tf.int32, [None, x_desc_dim], name="input_x_desc")
input_y = tf.placeholder(tf.float32, [None, y_dim], name="input_y")
labels = tf.placeholder(tf.int32, [None,5], name="Labels") 
pos_weight = tf.placeholder(tf.float32,[None, y_dim], name="pos_weight")
learning_rate = tf.placeholder(tf.float32, name="learning_rate")

is_train = FLAGS.is_train

word_dict = pd.read_pickle('./cache/word_dict.pkl')
word_embedding_matrix = np.load('./cache/trained_embedding_matrix.npy')


num_classes= y_dim
vocab_size=len(word_dict)
embedding_size=FLAGS.embedding_dim

with tf.device("/cpu:0"):
    Weights = tf.Variable(word_embedding_matrix, trainable=False, name="Weights")
    tf.summary.histogram('Embedding layer', Weights)

with tf.name_scope("sentence_embedding"):
    # 1. look up
    title_embedded_chars = tf.nn.embedding_lookup(Weights, input_x_title)
    desc_embedded_chars = tf.nn.embedding_lookup(Weights, input_x_desc)

    print "Shape of embedding_looked up",title_embedded_chars.shape

    # 2. average
    title_sentence_embeddings = tf.reduce_mean(title_embedded_chars, axis=1)
    desc_sentence_embeddings = tf.reduce_mean(desc_embedded_chars, axis=1)

    sentence_embeddings = tf.concat([title_sentence_embeddings, desc_sentence_embeddings],axis=1)

with tf.name_scope("drop_out"):
    keep_prob = FLAGS.drop_out_keep_prob
    h_flat = tf.reshape(sentence_embeddings,[-1, 2*embedding_dim])
    h_flat_drop = tf.nn.dropout(h_flat,keep_prob=keep_prob)

with tf.name_scope("classcifier"):
    # W：[256 * 1999]
    W_1 = tf.get_variable("classcifier/"+"W_1", [2*embedding_dim, hidden_unit],initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram('W_1', W_1)
    b_1 = tf.get_variable("classcifier/"+"b_1", [hidden_unit])
    tf.summary.histogram('b_1', b_1)

    W_2 = tf.get_variable("classcifier/"+"W_2", [hidden_unit, y_dim],initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram('W_2', W_2)
    b_2 = tf.get_variable("classcifier/"+"b_2", [y_dim])
    tf.summary.histogram('b_2', b_2)

    hidden = tf.matmul(h_flat_drop, W_1) + b_1
    bn = batch_norm_layer(hidden,train_phase=is_train,scope_bn="classcifier/bn")
    relu = tf.nn.relu(bn)
    


with tf.name_scope("logits"):
    logits = tf.matmul(relu, W_2) + b_2
    predict_top_5 = tf.nn.top_k(logits, k=5)
    

with tf.name_scope("score"):
    
    valid_score = tf.placeholder(tf.float32,name="score_eval")
    tf.summary.scalar('score_eval',valid_score)
    

with tf.name_scope("cross_entropy"):    
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_y,logits=logits))
    tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope("loss"):
    is_weighted = FLAGS.is_weighted
    if is_weighted:
        loss = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=input_y, pos_weight=pos_weight))
    else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_y,logits=logits)
        loss = tf.reduce_mean(loss)
    tf.summary.scalar('loss',loss)

#    l2_loss works bad. So we didn't use the l2 regularization.

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) 

with tf.name_scope("valid_loss"):
    valid_loss = tf.placeholder(tf.float32,name="valid_loss")
    tf.summary.scalar('valid_loss',valid_loss)
        
config = tf.ConfigProto(device_count={"CPU": 5}, # limit to num_cpu_core CPU usage  
                inter_op_parallelism_threads = 4,   
                intra_op_parallelism_threads = 5,  
                log_device_placement=True)
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

date = "0803_fastText"
model_save_path = "models/{}/model".format(date)
summary_save_path = "./summary/{}/".format(date)

adam_learning_rate = FLAGS.learning_rate

with tf.Session(config=config) as sess:
    label_top_5 = tf.nn.top_k(input_y, k=5) 
    sess.run(tf.global_variables_initializer())
    
    merged = tf.summary.merge_all()    
    train_writer = tf.summary.FileWriter(summary_save_path,sess.graph)  
    valid_writer = tf.summary.FileWriter(summary_save_path+"valid",sess.graph) 
    
    x_title_train, x_desc_train, y_train = load_data(model='train', shuffle=True)
    x_title_dev, x_desc_dev, y_dev = load_data(model='valid',shuffle=True)
    
    x_title_train = x_title_train[:,:x_title_dim]
    x_title_dev = x_title_dev[:,:x_title_dim]
    
    x_desc_train = x_desc_train[:,:x_desc_dim]
    x_desc_dev = x_desc_dev[:,:x_desc_dim]
    
    i = 0
    adam_learning_rate /= decay_rate 
        
    for epoch in range(FLAGS.num_epochs):
        
        adam_learning_rate *= decay_rate

        print("Shape of data -- train:{}\tvalid:{}".format(x_title_train.shape,x_title_dev.shape))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


        batches = batch_iter(
            list(zip(x_title_train, x_desc_train,  y_train)), FLAGS.batch_size, 1,shuffle=True)
        for batch in batches:
            i = i + 1
            x_title_batch, x_desc_batch, label_batch = zip(*batch)
            y_batch , y_batch_index = one_hot_coding(label_batch)
            loss_pos_weight = get_weights(label_batch,alpha=0.5, beta=0.35)
            
            sess.run([optimizer],feed_dict={input_x_title:x_title_batch, input_x_desc:x_desc_batch, input_y:y_batch,
                                            labels:y_batch_index,
                                                  pos_weight:loss_pos_weight,
                                            learning_rate:adam_learning_rate})
            
            if (i % FLAGS.evaluate_every == 0):
                train_predict_5, train_label_5, train_loss = sess.run([predict_top_5,label_top_5,loss],feed_dict={input_x_title:x_title_batch,
                                                                                                                 input_x_desc:x_desc_batch,
                                                                                                                input_y:y_batch,
                                                                                                                labels:y_batch_index,
                                                                                                                  pos_weight:loss_pos_weight,
                                                                                                                learning_rate:adam_learning_rate})
                train_predict_label_and_marked_label_list = []
                for predict,label in zip(train_predict_5[1],y_batch_index):
                    train_predict_label_and_marked_label_list.append((list(predict),list(label)))
                train_score = eval(train_predict_label_and_marked_label_list)
                
                print("Step {}:\ttrain loss:{}\t score:{}".format(i,train_loss, train_score))
                
                result = sess.run(merged,feed_dict={input_x_title:x_title_batch,
                                                    input_x_desc:x_desc_batch, 
                                                    input_y:y_batch,
                                                    labels:y_batch_index,
                                                    pos_weight:loss_pos_weight,
                                                    learning_rate:adam_learning_rate,
                                                    valid_score:train_score,
                                                    valid_loss:train_loss})
                train_writer.add_summary(result,i)
                
                if (i % (25* FLAGS.evaluate_every) == 0):
                    # 验证部分
                    valid_scores_list = []
                    valid_loss_list = []
                    print "Evaluation: at %d step:"%i

                    valid_batches = batch_iter(
                        list(zip(x_title_dev,x_desc_dev, y_dev)),FLAGS.batch_size, 1)
                    
                    valid_predict_label_and_marked_label_list = []
                    for valid_batch in valid_batches:
                        x_title_valid,x_desc_valid, label_valid = zip(*valid_batch)
                        y_valid,y_valid_index = one_hot_coding(label_valid)
                        loss_pos_weight = get_weights(label_valid,alpha=0.5, beta=0.35)
                        predict_5, label_5, _loss = sess.run([predict_top_5,label_top_5,loss],feed_dict={input_x_title:x_title_valid,
                                                                                                         input_x_desc:x_desc_valid,
                                                                                                         input_y:y_valid,
                                                                                                         labels:y_valid_index,
                                                                                                         pos_weight:loss_pos_weight,
                                                                                                         learning_rate:adam_learning_rate})
                        for predict,label in zip(predict_5[1],y_valid_index):
                            valid_predict_label_and_marked_label_list.append((list(predict),list(label)))
                        valid_loss_list.append(_loss)
                        
                    avg_valid_loss = sum(valid_loss_list) / len(valid_loss_list)
                    score = eval(valid_predict_label_and_marked_label_list)

                    print "loss:",_loss
                    print "score:",score
                    result = sess.run(merged,feed_dict={input_x_title:x_title_valid,
                                                        input_x_desc:x_desc_valid,
                                                        input_y:y_valid, 
                                                        labels:y_valid_index,
                                                        pos_weight:loss_pos_weight,
                                                        learning_rate:adam_learning_rate,
                                                        valid_score:score, 
                                                        valid_loss:avg_valid_loss})
                    valid_writer.add_summary(result,i)
            
            if (i % (FLAGS.checkpoint_every) == 0):
                path = saver.save(sess, model_save_path, global_step=i)
                print("Saved model checkpoint to {}".format(path))
