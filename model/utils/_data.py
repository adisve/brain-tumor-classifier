from utils._image import transform_image_file
from typing import List
import numpy as np
import os

def get_subfolders(directory) -> List:
    return [f.name for f in os.scandir(directory) if f.is_dir()]

def load_data(path, labels):
    X, y = [], []
    for label in labels:
        label_path = os.path.join(path, label)
        for file in os.listdir(label_path):
            image = transform_image_file(os.path.join(label_path, file))
            X.append(image)
            y.append(labels.index(label))
    return np.array(X) / 255.0, y
