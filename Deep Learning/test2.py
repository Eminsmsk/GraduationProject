import file_operations
import numpy as np
import matplotlib.pyplot as plt

import glob
data_path = 'all_datas/data/*'
mask_path = 'all_datas/mask/*'
data_names = glob.glob(data_path)
mask_names = glob.glob(mask_path)

data_names.sort()
mask_names.sort()

for i, j in zip(data_names, mask_names):

    data = np.load(i)
    mask = np.load(j)

    if mask.sum() > 0:
        d = np.hstack((data[:,:,2], mask*255))
        # plt.figure(1)
        # plt.imshow(data[:,:,0], cmap='gray')
        # plt.figure(2)
        # plt.imshow(mask, cmap='gray')
        plt.imshow(d, cmap='gray')

        plt.show()