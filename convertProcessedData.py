import numpy as np
np.set_printoptions(precision=2, suppress=True)

import os
import nibabel as nib
from nibabel.testing import data_path

import tensorflow as tf


def array_to_tfrecords(X, y, output_file):
    feature = {
        'x': tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten())),
        'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
        'shape' : tf.train.Feature(int64_list=tf.train.Int64List(value=X.shape))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()

    writer = tf.python_io.TFRecordWriter(output_file)
    writer.write(serialized)
    writer.close()


label = -1
for item_type in ["CTL", "ODN", "ODP"]:
    label += 1
    for i in range(1, 16):
        input_path = "/root/AUT/Project/Datasets/openfmri_parkinson/functional/%s%02d_test.feat/filtered_func_data.nii.gz" % (item_type, i)
        output_path = "/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_tfr_processed/%s%02d.tfrecord" % (item_type, i)

        example_ni1 = os.path.join(data_path, input_path)
        n1_img = nib.load(example_ni1)

        data = n1_img.get_data()

        m,n = data.shape[::2]
        inData = data.transpose(0,3,1,2).reshape(m,-1,n)

        array_to_tfrecords(inData, label, output_path)
