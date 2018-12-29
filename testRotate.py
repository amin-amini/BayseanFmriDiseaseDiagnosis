import numpy as np
np.set_printoptions(precision=2, suppress=True)

import matplotlib.pyplot as plt
import math

import os
import nibabel as nib
from nibabel.testing import data_path


for item_type in ["CTL", "ODN", "ODP"]:
    for i in range(1, 16):
        input_path = "/root/AUT/Project/Datasets/openfmri_parkinson/functional/%s%02d.nii.gz" % (item_type, i)
        output_path = "/root/AUT/Project/Datasets/openfmri_parkinson/ds000245_npy/%s%02d.npy" % (item_type, i)

        example_ni1 = os.path.join(data_path, input_path)
        n1_img = nib.load(example_ni1)

        data = n1_img.get_data()

        m,n = data.shape[::2]
        data_new = data.transpose(0,3,1,2).reshape(m,-1,n)

        # arr = np.array([i for i in range(0,255) for j in range(0,255)]).reshape(255,255) #data[:,:,0,0] * 255 / max(data_new.flatten())

        maxVal = max(data.flatten())

        plt.gray()
        fig = plt.figure(1)
        for t in range(0, data.shape[3]):
            arr = data[:,:,int(data.shape[2]/2),t] * 255.0 / maxVal

            plt.subplot(1, 3, 1)
            plt.imshow(arr)

            arr = data[:,int(data.shape[1]/2),:,t] * 255.0 / maxVal
            plt.subplot(1, 3, 2)
            plt.imshow(arr)

            arr = data[int(data.shape[0]/2),:,:,t] * 255.0 / maxVal
            plt.subplot(1, 3, 3)
            plt.imshow(arr)

            plt.show()
            print("test")

        exit(0)

