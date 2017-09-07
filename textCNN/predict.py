#coding=utf-8
import tensorflow as tf
import tqdm
import numpy as np
import re
from utils import *
from CNNmodel import TextCNN

tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 256)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("restore_path", "./models/0809/model-60000", "The model we're going to restore.")
tf.flags.DEFINE_string("model_name", "0809-CNN", "The name of the model.")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 256)")
tf.flags.DEFINE_integer("title_length", 40, "Length of title sequence")
tf.flags.DEFINE_integer("desc_length", 120, "Length of desc sequence")
tf.flags.DEFINE_integer("fc_nodes", 1024, "Number of nodes in the middle FC layer")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate (default: 1e-3)")

tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 8, "Number of training epochs (default: 10)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 200)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_bool('is_train', False, "True for training, False for inference")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

filter_size = map(int, FLAGS.filter_size.split(','))


def main(_):
    config = tf.ConfigProto(device_count={"CPU": 3},
                            inter_op_parallelism_threads=2,
                            intra_op_parallelism_threads=2,
                            log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        textCNN = TextCNN(name=FLAGS.model_name, filter_size=filter_size, feature_map=FLAGS.num_filters,
                          learning_rate=FLAGS.learning_rate, drop_out_keep_prob=FLAGS.dropout_keep_prob,
                          title_len=FLAGS.title_length, desc_len=FLAGS.desc_length, y_dim=1999,
                          fc_nodes=FLAGS.fc_nodes, evaluate_step=FLAGS.evaluate_every,
                          checkpoint_step=FLAGS.checkpoint_every, is_train=FLAGS.is_train)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, textCNN.checkpoint_path)

        x_title_test, x_desc_test = get_test_set()

        batches = batch_iter(list(zip(x_title_test, x_desc_test)), 1000, 1)
        predict_result = []
        counter = 1
        for batch in batches:
            x_title_batch, x_desc_batch = zip(*batch)

            predict_logits = sess.run(textCNN.logits,
                                      feed_dict={textCNN.input_x_title: x_title_batch,
                                                 textCNN.input_x_desc: x_desc_batch})
            for pred in predict_logits:
                predict_result.append(np.argsort(pred)[-1:-6:-1])

            if counter % 10 == 0:
                print ("Evaluation:step", counter)
            counter += 1

        pd.to_pickle(predict_result, "prediction.pkl")
        print "Result saved in ./prediction.pkl"

    predict_csv()


def get_test_set():
    # import question_train_set
    reader = pd.read_table('./ieee_zhihu_cup/question_eval_set.txt', sep='\t', header=None)
    print(reader.iloc[0:2])

    x_title_length = FLAGS.title_length
    x_desc_length = FLAGS.desc_length

    x_title_text = reader.iloc[:, 2]
    x_desc_text = reader.iloc[:, 2]

    word_dict = pd.read_pickle('./cache/word_dict.pkl')

    title_text = []
    for line in x_title_text:
        try:
            title_text.append(line.split(','))
        except:
            title_text.append(' ')

    desc_text = []
    for line in x_desc_text:
        try:
            desc_text.append(line.split(','))
        except:
            desc_text.append(' ')

    x_title = []
    for line in tqdm(title_text):
        line_len = len(line)
        text2num = []
        for i in xrange(x_title_length):
            if i < line_len:
                try:
                    text2num.append(word_dict[line[i]])
                except:
                    text2num.append(0)
            else:
                text2num.append(0)
        x_title.append(text2num)
    x_title = np.array(x_title)

    x_desc = []
    for line in tqdm(desc_text):
        line_len = len(line)
        text2num = []
        for i in xrange(x_desc_length):
            if i < line_len:
                try:
                    text2num.append(word_dict[line[i]])  # 把词转为数字
                except:
                    text2num.append(0)  # 没有对应的词
            else:
                text2num.append(0)  # 填充0
        x_desc.append(text2num)
    x_desc = np.array(x_desc)
    return x_title, x_desc


def predict_csv():
    topic_info = pd.read_table("./ieee_zhihu_cup/topic_info.txt", sep='\t', header=None)

    topic_dict = {}
    for i in xrange(topic_info.shape[0]):
        topic_dict[i] = topic_info.iloc[i][0]

    predict = pd.read_pickle("prediction.pkl")
    text = np.array(predict)

    label = []
    for line in tqdm(text):
        num2label = []
        for i in xrange(5):
            num2label.append(topic_dict[int(line[i])])
        label.append(num2label)
    label = np.array(label)
    np.savetxt("temp.txt", label, fmt='%d')

    def clean_str(string):
        string = re.sub(r" ", ",", string)
        return string

    file1 = open('temp.txt', "r")
    examples = file1.readlines()
    examples = [clean_str(line) for line in examples]
    file1.close()

    file1 = open('temp.txt', "w")
    file1.writelines(examples)
    file1.close()

    predict_file = 'temp.txt'
    predict_reader = pd.read_table(predict_file, sep=' ', header=None)

    eval_reader = pd.read_table('./ieee_zhihu_cup/question_eval_set.txt', sep='\t', header=None)

    final_predict = pd.concat([eval_reader.ix[:, 0], predict_reader], axis=1)
    final_predict.to_csv('temp.txt', header=None, index=None, sep=',')

    final_file = open('temp.txt', "r")
    final_examples = final_file.readlines()
    final_examples = [re.sub(r'"', "", line) for line in final_examples]
    final_file.close()

    final_file = open('final_predict.csv', "w")
    final_file.writelines(final_examples)
    final_file.close()
    print "Predict result saved in ./final_predict.csv"


if __name__ == '__main__':
    tf.app.run()
