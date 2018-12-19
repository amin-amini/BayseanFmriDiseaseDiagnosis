import numpy as np
np.set_printoptions(precision=2, suppress=True)

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

        np.save(output_path, data_new)
