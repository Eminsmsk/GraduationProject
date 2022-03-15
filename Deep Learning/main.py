import file_operations as f
import patch_extractor as pe
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy

if __name__ == '__main__':
    patch_size = 512
    building_path = 'buildings.tiff'
    color_path = 'ODTU_color/color.bip'
    ndsm_path = 'ndsm_file.tiff'
    ndsm_data = f.read(ndsm_path, patch_size, 'height', False, True, True, True)
    color_data = f.read(color_path, patch_size, None, True, False, False, True)
    building_mask = f.read(building_path, patch_size, 'mask', False, False, False, True)
    building_mask[building_mask > 0] = 1

    pe.extract(color_data, ndsm_data, building_mask, patch_size)

