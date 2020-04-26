import os
import numpy as np
from PIL import Image as img
from random import shuffle

def load_data(dir, width=224, height=224):
    global CLASS_NAMES
    CLASS_NAMES = [item for item in os.listdir(dir) if item != "LICENSE.txt"]
    class_dirs = np.array([dir+"/"+item for item in CLASS_NAMES])
    lst_dirs = [os.listdir(item) for item in class_dirs]
    lst_dirs = [class_dirs[d]+"/"+item for d in range(len(lst_dirs)) for item in lst_dirs[d]]
    shuffle(lst_dirs)
    x_train = np.zeros((len(lst_dirs), width, height, 3), dtype=np.float32)
    y_train = np.zeros((len(lst_dirs), len(CLASS_NAMES),), dtype=np.float32)
    for i_path in range(len(lst_dirs)):
        x_train[i_path] = decode_img(lst_dirs[i_path], width, height)
        y_train[i_path] = get_label(lst_dirs[i_path])
    return x_train, y_train


def decode_img(file, width, height):
    i = img.open(file)
    i = i.resize((width, height))
    return np.asarray(i, dtype="float32")

def get_label(path):
    res = np.zeros((5,))
    res[CLASS_NAMES.index(path.split("/")[-2])] = 1.0
    return res