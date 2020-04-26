import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

def main():
    if len(sys.argv) < 3:
        print("Usage: {} <image file> <model file>".format(sys.argv[0]))
        sys.exit()
    model = tf.keras.models.load_model(sys.argv[2])
    img = prepare_image(sys.argv[1])
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = img / 255.0
    pred = model.predict_proba(img, verbose=2)
    print("{}:          {}%".format(CLASS_NAMES[0], pred[0][0]*100))
    print("{}:      {}%".format(CLASS_NAMES[1], pred[0][1]*100))
    print("{}:           {}%".format(CLASS_NAMES[2], pred[0][2]*100))
    print("{}:      {}%".format(CLASS_NAMES[3], pred[0][3]*100))
    print("{}:          {}%".format(CLASS_NAMES[4], pred[0][4]*100))



def prepare_image(file, target_size=(224,224)):
    img = Image.open(file)
    img = img.resize(target_size)
    return np.asarray(img, dtype="float32")

main()
