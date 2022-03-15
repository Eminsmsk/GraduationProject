import tensorflow as tf
from sklearn.model_selection import train_test_split
import dataset as D
import data_loader as DL
import model as M
import os
import numpy as np
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    datas, masks = D.get_splitted_datas()

    train_generator = DL.get_generator(datas, masks, 4)








    model = M.unet()
    model.fit(train_generator, epochs=500, verbose=1, shuffle=False)
    # model.save('record9/')


