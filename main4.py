from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import shuffle
import math

import numpy as np
import tensorflow as tf

from subprocess import call

# call(["rm", "-rf", "/root/AUT/Project/Codes/firstTry/outmodel2"])
# call(["mkdir", "/root/AUT/Project/Codes/firstTry/outmodel2"])

ITERATIONS = 50

tf.logging.set_verbosity(tf.logging.DEBUG)

trainRange = range(1, 13)
trainTimeRange = range(0,159)
validationTimeRange = range(159,198)

testRange = range(13, 16)

formatPath = "/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_tfr_expanded/%s%02d_%03d.tfrecord"

trainPath = [formatPath % (c, i, t) for c in ["CTL", "ODP"] for i in trainRange for t in trainTimeRange]
validationPath = [formatPath % (c, i, t) for c in ["CTL", "ODP"] for i in trainRange for t in validationTimeRange]
testPath = [formatPath % (c, i, t) for c in ["CTL", "ODP"] for i in testRange for t in range(0,198)]

# note that shuffling is not needed for test data and ordered data makes test code a bit easier to write
shuffle(trainPath)
shuffle(validationPath)


def extract_fn(data_record):
    features = {
        # Extract features using the keys set during creation
        'x': tf.FixedLenFeature([159744], tf.float32),
        'y': tf.FixedLenFeature([1], tf.int64),
        'shape': tf.FixedLenFeature([3], tf.int64)
    }
    sample = tf.parse_single_example(data_record, features)

    return sample, tf.minimum(sample['y'], 1)


def bayesian_cnn_model_fn(features, labels, mode):
    """Model function for Bayesian CNN."""
    # Input Layer
    with tf.Session() as sess:
        shape = tf.Tensor.eval(features["shape"], session=sess)

    input_layer = tf.reshape(features["x"],
                             [-1, shape[0], shape[1], shape[2], 1])


    # Convolutional Layer #1
    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=4,
        kernel_size=[5, 5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Dropout Layer #1
    dropout1 = tf.layers.dropout(inputs=conv1, rate=0.05, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling3d(inputs=dropout1, pool_size=[4, 4, 4], strides=4)

    # Convolutional Layer #2
    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=8,
        kernel_size=[4, 4, 4],
        padding="same",
        activation=tf.nn.relu)

    # Dropout Layer #2
    dropout2 = tf.layers.dropout(inputs=conv2, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Pooling Layer #2
    pool2 = tf.layers.max_pooling3d(inputs=dropout2, pool_size=[4, 4, 4], strides=4)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 2 * 8])
    dense = tf.layers.dense(inputs=pool2_flat, units=16, activation=tf.nn.relu)

    # final Dropout Layer
    dropout = tf.layers.dropout(inputs=dense, rate=0.15, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    tiledLabels = tf.tile(labels, [input_layer.shape[0]])

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tiledLabels, logits=logits)

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
            labels=tiledLabels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def file_input_fn(filenames):
    dataset = tf.data.TFRecordDataset(filenames).map(extract_fn)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def dataset_input_fn(mode, iteration):
    file_names = {
        tf.estimator.ModeKeys.TRAIN: trainPath,
        tf.estimator.ModeKeys.EVAL: validationPath
    }[mode]

    l = len(file_names)
    size = math.floor(l/ITERATIONS)

    start_index = iteration*size
    end_index = min(start_index + size, l)

    return file_input_fn(file_names[start_index:end_index])


def train_input_fn(iteration):
    return dataset_input_fn(tf.estimator.ModeKeys.TRAIN, iteration)


def eval_input_fn(iteration):
    return dataset_input_fn(tf.estimator.ModeKeys.EVAL, iteration)


def test_input_fn(filename):
    dataset = tf.data.TFRecordDataset([filename]).map(extract_fn)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def main(unused_argv):
    # model = (bayesian_cnn_model_fn, "./bayesian_cnn_model_fn")
    model = (bayesian_cnn_model_fn, "./outmodel9_expand_processed")

    my_classifier = tf.estimator.Estimator(model_fn=model[0], model_dir=model[1])

    tensors_to_log = {
        "probabilities": "softmax_tensor"
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)

    for iteration in range(0, ITERATIONS):
        my_classifier.train(
            input_fn=lambda: train_input_fn(iteration),
            hooks=[logging_hook])

        eval_results = my_classifier.evaluate(input_fn=lambda: eval_input_fn(iteration))
        print(eval_results)

    predict_results = my_classifier.predict(input_fn=test_input_fn)
    print(predict_results)

    PrintBuffer = []

    correctCount = 0
    totalCount = 0

    label = -1
    for cls in ["CTL", "ODP"]:
        label += 1
        for instance in testRange:
            predicted_labels = [0, 0]
            for t in range(0, 198):
                predict_results = my_classifier.predict(input_fn=lambda: test_input_fn(formatPath % (cls, instance, t)))
                for prediction in predict_results:
                    predicted_label = prediction['classes']
                    predicted_labels[predicted_label] = predicted_labels[predicted_label] + 1

            prediction = 0 if predicted_labels[0] > predicted_labels[1] else 1

            if prediction == label:
                correctCount = correctCount + 1
            totalCount = totalCount+1

            PrintBuffer.append( "result : %s %02d  :  %d , %d" % (cls , instance ,  predicted_labels[0] , predicted_labels[1] ) )
            print('result : ', cls, instance, predicted_labels)

    for pb in PrintBuffer:
        print(pb)
    print('accuracy is %f' % ( 1.0 * correctCount / totalCount ))

    print("test")


if __name__ == "__main__":
    tf.app.run()
