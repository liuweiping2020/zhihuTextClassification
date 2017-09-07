# coding=utf-8
import tensorflow as tf
import numpy as np
import math
import pandas as pd


def load_data(model="whole", shuffle=False):
    """
    :param model:
        str, 'whole'：加载完整数据集，'train':加载训练集, 'validation':加载验证集
    :param shuffle:
        boolean, True for shuffle
    :return:
        title_data: title数据
        desc_data： description数据
        y： 数据标签
    """
    print "Loading " + model + " data."

    if model == "whole":
        title_path = './cache/train_data_title.npy'
        desc_path = './cache/train_data_desc.npy'
        label_path = './cache/train_label.csv'

        print("Loading title  data ...")
        title_data = np.load(title_path)
        print("Loading description  data ...")
        desc_data = np.load(desc_path)
        print ("Loading label...")
        y = pd.read_csv(label_path, header=None)
        y = np.array(y.iloc[:, 1])

    if model == "train":
        title_path = './cache/train/new_train_title.npy'
        desc_path = './cache/train/new_train_desc.npy'
        label_path = './cache/train/new_y_train.npy'

        print("Loading train title  data ...")
        title_data = np.load(title_path)
        print("Loading train description  data ...")
        desc_data = np.load(desc_path)
        print ("Loading train label...")
        y = np.load(label_path)

    if model == "validation":
        title_path = './cache/valid/new_valid_title.npy'
        desc_path = './cache/valid/new_valid_desc.npy'
        label_path = './cache/valid/new_y_valid.npy'

        print("Loading validation title  data ...")
        title_data = np.load(title_path)
        print("Loading validation description  data ...")
        desc_data = np.load(desc_path)
        print ("Loading validation label...")
        y = np.load(label_path)

    data_size = y.shape[0]

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        title_data = title_data[shuffle_indices]
        desc_data = desc_data[shuffle_indices]
        y = y[shuffle_indices]

    return title_data, desc_data, y


def one_hot_coding(y):
    """
    由于label维度有1999维，因此占用空间非常大
    所以将label转化为one-hot 的部分单独拿出来
    :param y:
        str, 所属类别的id,like ['14,7,34','2,34,634,566,'1493']
    :return:
        one_hot_labels , ndarray, one-hot形式的label
        index_labels , list, label类别的编号,like [[14,7,34], [2,34,634,566,1493]]
    """
    one_hot_labels = []
    index_labels = []
    for i in range(len(y)):
        temp = y[i].split(',')
        # 如果分类数大于5，只取前5个分类
        if (len(temp) > 5):
            temp = temp[0:5]
        # 设置标签的对应位置为1，其余位置为0
        label = np.zeros(1999)
        index_label = []
        for j, temp_label in enumerate(temp):
            label[int(temp_label)] = 1
            index_label.append(int(temp_label))
        one_hot_labels.append(label)
        index_labels.append(index_label)
    one_hot_labels = np.array(one_hot_labels, dtype=np.float32)
    return one_hot_labels, index_labels


def get_weights(pos, alpha=1.0, beta=0.0):
    """
    获取标签的按位置排列的权重
    :param pos:
        list, like [2,4,1,9,5]
    :param alpha:
        float, 控制加权的程度，alpha=0时权重为全1,不产生加权效果
    :param beta:
        float, 控制加权的偏置
    :return:
        加权向量  shape=(1999,)， 某一位置的加权值为  1 + alpha*(pos_weight[i]-1+beta)
    """
    length = len(pos)
    weights = []
    e = math.e
    for k in range(length):
        temp = pos[k].split(',')
        num_label = len(temp)
        if len(temp) > 5:
            temp = temp[0:5]
            num_label = 5
        # 计算权重
        weight = np.ones(1999)
        sigma = 0
        for i in range(len(temp)):
            sigma += math.log(e, i + 2)
        for i, p in enumerate(temp):
            w = math.log(e, i + 2)
            weight[int(p)] = w * num_label / sigma + beta
        weights.append(weight)
    weights = np.array(weights, dtype=np.float32)
    # 减小加权的程度
    ones = np.ones_like(weights)
    weights = alpha * weights + (1.0 - alpha) * ones

    return weights


# 生成批次数据
def batch_iter(data, batch_size, num_epochs=1, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    # 每个epoch的num_batch
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("num_batches_per_epoch:", num_batches_per_epoch)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            np.random.shuffle(data)

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]


def eval(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)
    """
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  # 在各个位置上总命中数量
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num
    if precision * recall == 0:
        return 0
    return (precision * recall) / (precision + recall)


def eval_logits(logits, y_index):
    predict_results = []
    predict_label_and_marked_label_list = []
    for logit in logits:
        predict_results.append(list(np.argsort(logit)[-1:-6:-1]))
    for predict, label in zip(predict_results, y_index):
        predict_label_and_marked_label_list.append((list(predict), list(label)))
    score = eval(predict_label_and_marked_label_list)
    return score


def batch_norm_layer(x, train_phase, scope_bn):
    """
    Batch Normalization
    Copy from https://www.zhihu.com/question/53133249
    :param x:
        Tensor, 输入张量
    :param train_phase:
        boolean, 'True' for training, 'False' for inference
    :param scope_bn:
        str, Name scope
    :return normed：
        Tensor, Batch Normalized.
    """
    train_phase = np.bool_(train_phase)
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = list(np.arange(len(x.shape) - 1))
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed