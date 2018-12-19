from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import shuffle

# Imports
import numpy as np
import tensorflow as tf

DUPLICATION = 1

tf.logging.set_verbosity(tf.logging.INFO)


def extract_fn(data_record):
    features = {
        # Extract features using the keys set during creation
        'x': tf.FixedLenFeature([31629312], tf.float32),
        'y': tf.FixedLenFeature([1], tf.int64),
        'shape': tf.FixedLenFeature([3], tf.int64)
    }
    sample = tf.parse_single_example(data_record, features)

    return sample , sample['y']


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    with tf.Session() as sess:
        shape = inputShape = tf.Tensor.eval(features["shape"], session= sess)

    input_layer = tf.reshape(features["x"],
                             [-1, shape[0], shape[1], shape[2], 1])


    preePool = tf.layers.average_pooling3d(inputs=input_layer, pool_size=[1, 128, 1], strides=[1, 128, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv3d(
        inputs=preePool,
        filters=4,
        kernel_size=[5, 5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[8, 8, 8], strides=8)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=8,
        kernel_size=[4, 4, 4],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[4, 4, 4], strides=4)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 2 * 3 * 1 * 8])
    dense = tf.layers.dense(inputs=pool2_flat, units=16, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def getTfPath():
    return ["/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_tfr/%s%02d.tfrecord" % (item_type, i) for item_type
            in ["CTL", "ODN", "ODP"] for i in range(1, 16)]

tfPaths = getTfPath()
shuffle(tfPaths)

def dataset_input_fn(train=True):
    filenames = tfPaths[0:30] * DUPLICATION if train else tfPaths[30:45]

    dataset = tf.data.TFRecordDataset(filenames ).map(extract_fn)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def train_input_fn():
    return dataset_input_fn(train=True)


def test_input_fn():
    return dataset_input_fn(train=False)


def main(unused_argv):
    # dataset = tf.data.TFRecordDataset(getTfPath()).map(extract_fn)
    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     try:
    #         # Keep extracting data till TFRecord is exhausted
    #         while True:
    #             sample_data = sess.run(next_element)
    #
    #             print('Save path ')
    #     except:
    #         pass

    my_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./outmodel")

    tensors_to_log = {
        "probabilities": "softmax_tensor",
        "loss": "loss"
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)

    my_classifier.train(
        input_fn=train_input_fn,
        #steps=2000,
        hooks=[logging_hook])

    eval_results = my_classifier.evaluate(input_fn=test_input_fn)
    print(eval_results)

    print("test")


if __name__ == "__main__":
    tf.app.run()
