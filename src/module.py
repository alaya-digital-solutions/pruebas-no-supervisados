import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint, random


def get_data(path_data, max_data, dsize):
    X = []
    for root, dirs, files in os.walk(path_data):
        # print("root", root)
        # print("dirs", dirs)
        # print("files", files)
        if files is not None:
            print("no esta vacio")
            for filename in files:
                path = path_data + filename
                img = cv2.imread(path, 0)
                # print(path)
                if img is None:
                    continue
                if random() < max_data/len(files):
                    img = cv2.resize(img, dsize)
                    if random() < 0.5:
                        img = cv2.flip(img, randint(0, 1))
                    X.append(img)
                if len(X) >= max_data:
                    return X
    return X


def plot_n(activos, n=2):
    for i in range(2):
        plt.imshow(activos[randint(0, len(activos))], cmap='gray')
        plt.show()


def image_normalization(activos, img_height, img_width):
    activos = np.array(activos, dtype=np.uint8)
    activos = activos.reshape(activos.shape[0], img_height, img_width)
    activos = activos.astype('float32')
    activos /= 255
    return activos
