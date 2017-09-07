# coding=utf-8
import tensorflow as tf
from CNNmodel import TextCNN
from utils import *

tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 256)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("model_name", "0809-CNN", "The name of the model")
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

tf.flags.DEFINE_bool('is_train', True, "True for training, False for inference")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))


def main(_):
    config = tf.ConfigProto(device_count={"CPU": 3},
                            inter_op_parallelism_threads=2,
                            intra_op_parallelism_threads=2,
                            log_device_placement=True)

    config.gpu_options.allow_growth = True

    filter_size = map(int, FLAGS.filter_size.split(','))
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    with tf.Session(config=config) as sess:
        textCNN = TextCNN(name=FLAGS.model_name, filter_size=filter_size, feature_map=FLAGS.num_filters,
                          learning_rate=FLAGS.learning_rate, drop_out_keep_prob=FLAGS.dropout_keep_prob,
                          title_len=FLAGS.title_length, desc_len=FLAGS.desc_length, y_dim=1999,
                          fc_nodes=FLAGS.fc_nodes,evaluate_step=FLAGS.evaluate_every,
                          checkpoint_step=FLAGS.checkpoint_every, is_train=FLAGS.is_train)

        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(textCNN.summary_path, sess.graph)
        valid_writer = tf.summary.FileWriter(textCNN.summary_path+"validation", sess.graph)

        x_title_train, x_desc_train, y_train = load_data(model='train', shuffle=True)
        x_title_dev, x_desc_dev, y_dev = load_data(model='valid', shuffle=False)

        x_title_train = x_title_train[:, :textCNN.title_len]
        x_title_dev = x_title_dev[:, :textCNN.title_len]
        x_desc_train = x_desc_train[:, :textCNN.desc_len]
        x_desc_dev = x_desc_dev[:, :textCNN.desc_len]

        global_step = sess.run(textCNN.global_step)
        for epoch in range(FLAGS.num_epochs):
            print("Training Title Data shape: {},{}".format(x_title_train.shape, x_desc_train.shape))
            print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

            batches = batch_iter(
                list(zip(x_title_train, x_desc_train, y_train)), FLAGS.batch_size, shuffle=True)

            for batch in batches:
                x_title_batch, x_desc_batch, label_batch = zip(*batch)
                y_batch, y_index_batch = one_hot_coding(label_batch)

                # train
                sess.run(textCNN.train_op, feed_dict={textCNN.input_x_title: x_title_batch,
                                                      textCNN.input_x_desc: x_desc_batch,
                                                      textCNN.input_y: y_batch})

                if global_step % FLAGS.evaluate_every == 0:
                    train_logits, train_loss = sess.run([textCNN.logits, textCNN.loss],
                                                        feed_dict={textCNN.input_x_title: x_title_batch,
                                                                   textCNN.input_x_desc: x_desc_batch,
                                                                   textCNN.input_y: y_batch})
                    train_score = eval_logits(train_logits, y_index_batch)
                    print("Step {}:\ttrain loss:{}\t score:{}".format(global_step, train_loss, train_score))

                    result = sess.run(merged, feed_dict={textCNN.input_x_title: x_title_batch,
                                                         textCNN.input_x_desc: x_desc_batch,
                                                         textCNN.input_y: y_batch,
                                                         textCNN.valid_score: train_score,
                                                         textCNN.valid_loss: train_loss})
                    train_writer.add_summary(result, global_step)

                    # validation
                    if global_step % FLAGS.evaluate_every == 0:
                        print "Evaluation: at %d step:" % global_step

                        valid_batches = batch_iter(
                            list(zip(x_title_dev, x_desc_dev, y_dev)), FLAGS.batch_size, 1)

                        validation_scores = []
                        validation_losses = []
                        for valid_batch in valid_batches:
                            x_title_valid, x_desc_valid, label_valid = zip(*valid_batch)
                            y_valid, y_index_valid = one_hot_coding(label_valid)
                            valid_logits, _loss = sess.run(
                                [textCNN.logits, textCNN.loss],
                                feed_dict={textCNN.input_x_title: x_title_valid,
                                           textCNN.input_x_desc: x_desc_valid,
                                           textCNN.input_y: y_valid})
                            validation_scores.append(eval_logits(valid_logits, y_index_valid))
                            validation_losses.append(_loss)
                        avg_validation_score = sum(validation_scores)/len(validation_scores)
                        avg_validation_loss = sum(validation_losses)/len(validation_losses)
                        print("Validation {}:\t loss:{}\t score:{}".format(
                            global_step, avg_validation_loss, avg_validation_score))

                        result = sess.run(merged,
                                          feed_dict={textCNN.input_x_title: x_title_valid,
                                                     textCNN.input_x_desc: x_desc_valid,
                                                     textCNN.input_y: y_valid,
                                                     textCNN.valid_score: avg_validation_score,
                                                     textCNN.valid_loss: avg_validation_loss})
                        valid_writer.add_summary(result, global_step)

            if global_step % FLAGS.checkpoint_every == 0:
                # Only save the models which have the score>0.39
                if train_score >= 0.39:
                    path = saver.save(sess, textCNN.checkpoint_path, global_step=global_step)
                    print("Saved model checkpoint to {}".format(path))


if __name__ == '__main__':
    tf.app.run()