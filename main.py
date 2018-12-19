from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, features["x"].shape[1], features["x"].shape[2], features["x"].shape[3], 1])

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
  pool2_flat = tf.reshape(pool2, [-1, 2*3*1*8 ])
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


def main(unused_argv):
  train_data = np.array([
    np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/CTL01.npy"),
    #np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/CTL02.npy"),
    #np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/CTL03.npy"),
    np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/ODN01.npy"),
    #np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/ODN02.npy"),
    #np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/ODN03.npy"),
    np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/ODP01.npy"),
    #np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/ODP02.npy"),
    #np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/ODP03.npy")
  ])
  train_data = train_data.astype(np.float32, order='C') / train_data.max()
  #train_labels = np.asarray([1,1,1,2,2,2,3,3,3], dtype=np.int32)
  train_labels = np.asarray([1,2,3], dtype=np.int32)
  eval_data = np.array([
    np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/CTL04.npy"),
    np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/ODN04.npy"),
    np.load("/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/ODP04.npy")
  ])
  eval_data = eval_data.astype(np.float32, order='C') / train_data.max()
  eval_labels = np.asarray([1,2,3], dtype=np.int32)

  my_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="./outmodel")

  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=2,
    num_epochs=None,
    shuffle=True)
  my_classifier.train(
    input_fn=train_input_fn,
    steps= 2, #20,
    hooks=[logging_hook])

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  eval_results = my_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

  print ("test")
if __name__ == "__main__":
  tf.app.run()
