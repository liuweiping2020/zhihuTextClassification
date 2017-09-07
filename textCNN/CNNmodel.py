# coding:utf-8
import tensorflow as tf
from utils import *


class TextCNN(object):
    def __init__(self, name="textCNN", filter_size=[3, 4, 5], feature_map=256,
                 embedding_size=256, learning_rate=0.001, drop_out_keep_prob=0.5,
                 title_len=40, desc_len=120, y_dim=1999, fc_nodes=1024,
                 evaluate_step=200, checkpoint_step=2000, is_train=True):
        """
        Args:

        :param name: Name of the model.
        :param filter_size: window size like [3,4,5].
        :param feature_map: feature map of the kernel .
        :param embedding_size: size of embedding vector.
        :param learning_rate: learning rate of the optimizer.
        :param drop_out_keep_prob: keep probability of drop out layer.
        :param title_len: input length of title data.
        :param desc_len: input length of desc data.
        :param y_dim: class number.
        :param fc_nodes: how many nodes in the middle FC layer.
        :param evaluate_step: how many steps when the model evaluate itself.
        :param checkpoint_step: how many steps when the model save checkpoint.
        :param is_train: boolean, "True" for train, "False" for inference.
        """

        self.name = name
        self.filter_size = filter_size
        self.feature_map = feature_map
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.drop_out_keep_prob = drop_out_keep_prob
        self.title_len = title_len
        self.desc_len = desc_len
        self.y_dim = y_dim
        self.fc_nodes = fc_nodes
        self.evaluate_step = evaluate_step
        self.checkpoint_step = checkpoint_step
        self.is_train = is_train
        self.summary_path = './summary/'+self.name+'/'
        self.checkpoint_path = './ckpt/'+self.name+'/'
        self.global_step = tf.Variable(0, trainable=False, name='Global_Step')

        with tf.name_scope("Input"):
            self.input_x_title = tf.placeholder(tf.int32, [None, self.title_len], name="input_x_title")
            self.input_x_desc = tf.placeholder(tf.int32, [None, self.desc_len], name="input_x_desc")
            self.input_y = tf.placeholder(tf.float32, [None, self.y_dim], name="input_y")

        with tf.name_scope("Embedding"):
            self.Weights = tf.Variable(np.load('./cache/trained_embedding_matrix.npy'), trainable=True, name="Weights")
            self.embedded_words_title = tf.nn.embedding_lookup(self.Weights, self.input_x_title)
            self.embedded_words_title_expanded = tf.expand_dims(self.embedded_words_title, -1)
            self.embedded_words_desc = tf.nn.embedding_lookup(self.Weights, self.input_x_desc)
            self.embedded_words_desc_expanded = tf.expand_dims(self.embedded_words_desc, -1)
            tf.summary.histogram('Embedding layer', self.Weights)

        with tf.name_scope("score"):
            valid_score = tf.placeholder(tf.float32, name="score")
            tf.summary.scalar('score', valid_score)

        with tf.name_scope("valid_loss"):
            valid_loss = tf.placeholder(tf.float32, name="valid_loss")
            tf.summary.scalar('valid_loss', valid_loss)

        self.pos_weight = tf.placeholder(tf.float32, [None, self.y_dim], name="pos_weight")

        self.logits = self.inference()
        self.loss = self.loss()
        self.train_op = self.train()

    def inference(self):
        pooled_title_outputs = []
        with tf.name_scope("title"):
            for i, filter_size in enumerate(self.filter_size):
                with tf.name_scope("conv-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, self.feature_map]
                    W_1 = tf.Variable(
                        tf.truncated_normal(filter_shape, stddev=0.1), name="W_1")
                    tf.summary.histogram("W_1", W_1)

                    b_1 = tf.Variable(
                        tf.constant(0.1, shape=[self.feature_map]), name="b_1")
                    tf.summary.histogram("b_1", b_1)

                    conv = tf.nn.conv2d(
                        self.embedded_words_title_expanded,
                        W_1,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    conv = tf.nn.bias_add(conv, b_1)

                    bn = batch_norm_layer(conv, train_phase=self.is_train,
                                          scope_bn="title/conv-%s/conv_bn" % filter_size)

                    # Apply nonlinearity
                    relu = tf.nn.relu(bn, name="relu")
                    shape = relu.shape.as_list()
                    h = tf.reshape(relu, [-1, shape[1], shape[3], shape[2]])
                    tf.summary.histogram("relu", h)

                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.title_len - (filter_size - 1), 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    tf.summary.histogram("pool", pooled)
                    pooled_title_outputs.append(pooled)

        pooled_desc_outputs = []
        with tf.name_scope("description"):
            for i, filter_size in enumerate(self.filter_size):
                with tf.name_scope("conv-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_size, 1, self.feature_map]
                    W_1 = tf.Variable(
                        tf.truncated_normal(filter_shape, stddev=0.1), name="W_1")
                    tf.summary.histogram("W_1", W_1)

                    b_1 = tf.Variable(
                        tf.constant(0.1, shape=[self.feature_map]), name="b_1")
                    tf.summary.histogram("b_1", b_1)

                    conv = tf.nn.conv2d(
                        self.embedded_words_desc_expanded,
                        W_1,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv_1")
                    conv = tf.nn.bias_add(conv, b_1)

                    bn = batch_norm_layer(conv, train_phase=self.is_train,
                                          scope_bn="description/conv-%s/conv_bn" % filter_size)

                    # Apply nonlinearity
                    relu = tf.nn.relu(bn, name="relu")
                    shape = relu.shape.as_list()
                    h_desc = tf.reshape(relu, [-1, shape[1], shape[3], shape[2]])
                    tf.summary.histogram("relu", h_desc)

                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h_desc,
                        ksize=[1, self.desc_len - (filter_size - 1), 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    tf.summary.histogram("pool", pooled)

                    pooled_desc_outputs.append(pooled)

        pooled_outputs = pooled_title_outputs + pooled_desc_outputs
        num_filters_total = 2 * self.feature_map * len(self.filter_size)

        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        tf.summary.histogram("Concated_Flatten", h_pool_flat)

        with tf.name_scope("Drop_out"):
            h_pool_flat_dropout = tf.nn.dropout(h_pool_flat, self.drop_out_keep_prob)

        with tf.name_scope("FC"):
            # FC layer 1
            fc_W_1 = tf.get_variable(
                "FC/W_1",
                shape=[num_filters_total, self.fc_nodes],
                initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_1', fc_W_1)
            fc_b_1 = tf.Variable(tf.constant(0.1, shape=[self.fc_nodes]), name="b_1")
            tf.summary.histogram('b_1', fc_b_1)

            bn_1 = batch_norm_layer(tf.matmul(h_pool_flat_dropout, fc_W_1, name="fc_hidden") + fc_b_1,
                                    train_phase=self.is_train, scope_bn='bn_1')
            hidden = tf.nn.relu(bn_1)

            # FC layer 2
            fc_W_2 = tf.get_variable(
                "FC/W_2",
                shape=[self.fc_nodes, self.y_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_2', fc_W_2)

            fc_b_2 = tf.Variable(tf.constant(0.1, shape=[self.y_dim]), name="b_2")
            tf.summary.histogram('b_2', fc_b_2)

        with tf.name_scope("logits"):
            logits = tf.nn.xw_plus_b(hidden, fc_W_2, fc_b_2, name="logits")
            tf.summary.histogram('logits', logits)
        return logits

    def loss(self):
        with tf.name_scope("cross_entropy_loss"):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            cross_entropy_loss = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_loss
        tf.summary.scalar('loss', loss)

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam")
        return train_op