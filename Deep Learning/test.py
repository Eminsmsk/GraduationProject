import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from skimage import io, util
path = 'ODTU_color/color.bip'

# building_mask = io.imread('buildings.tiff').astype(np.uint8)
# building_mask[building_mask > 0] = 1
#
# tree_mask = io.imread('trees.tiff').astype(np.uint8)
# tree_mask[tree_mask > 0] = 1


dataset = gdal.Open(path)

if dataset == None:
    print("File cannot be opened")

im_width = dataset.RasterXSize # The number of columns of the raster matrix
im_height = dataset.RasterYSize # Number of rows of the raster matrix
im_bands = dataset.RasterCount # number of bands

print(im_width, im_height, im_bands)
im_data = dataset.ReadAsArray(0, 0, im_width, im_height) # Get data
im_data = im_data.transpose((1,2,0))

import cv2

red = im_data[:500,:500,0].astype('f4')
green = im_data[:500,:500,1].astype('f4')

ndvi_nan = np.divide(np.subtract(green, red), np.add(green, red))
ndvi = np.nan_to_num(ndvi_nan, nan=-1, posinf=-1, neginf=-1)
ndvi = cv2.normalize(ndvi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
ndvi = ndvi.astype(np.uint8)

print(ndvi.max(), ndvi.min())
print(red.max(), red.min())
print(green.max(), green.min())

plt.figure(1)
plt.imshow(ndvi, cmap='gray')

plt.figure(2)
plt.imshow(red, cmap='gray')



plt.show()



# for i in ndvi.flatten():
#     print(i)

print(ndvi.shape)



del dataset
