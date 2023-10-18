import tensorflow as tf
import cv2
from enum import Enum


class Classifier(str, Enum):
    DenseNet = "DenseNet"
    EfficientNet = "EfficientNet"
    MobileNet = "MobileNet"
    ResNet = "ResNet"


classifier_map = {
    "DenseNet": "DenseNet.h5",
    "EfficientNet": "EfficientNet.h5",
    "MobileNet": "MobileNet.h5",
    "ResNet": "ResNet50.h5",
}


def load_brain_tumor_classifier(c_key: Classifier):
    match classifier_map.get(c_key):
        case link if link is not None:
            return tf.keras.models.load_model("../model/" + link)

        case _:
            return "Classifier Model not found"


def augment_image_cv2(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(image, 2, 50, 50)
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    image = cv2.resize(image, (200, 200))
    return image
