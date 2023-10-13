from tensorflow import keras
import cv2

class BrainTumorClassifier():
    def __init__(self) -> None:
        self.model = self.load_model(model_path='../model/model.h5')

    def load_model(self, model_path):
        try:
            return keras.models.load_model(model_path)
        except OSError:
            return None


def augment_image_cv2(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(image, 2, 50, 50)
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    image = cv2.resize(image, (200, 200))
    return image