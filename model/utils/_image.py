import matplotlib.pyplot as plt
import os
import cv2

def display_images_from_subfolders(training_dir, subfolders):
    plt.figure(figsize=(12, 8))
    for i, subfolder in enumerate(subfolders):
        image = __get_first_image_from_subfolder(os.path.join(training_dir, subfolder))
        if image is not None:
            plt.subplot(1, len(subfolders), i + 1)
            plt.title(subfolder)
            plt.axis('off')
            plt.imshow(image)
    plt.show()

def transform_image_file(file, image_size=224):
    image = cv2.imread(file, 0) 
    image = cv2.bilateralFilter(image, 2, 50, 50)
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    image = cv2.resize(image, (image_size, image_size))
    return image

def __get_first_image_from_subfolder(subfolder_path):
    image_files = os.listdir(subfolder_path)
    if image_files:
        image_path = os.path.join(subfolder_path, image_files[0])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    return None
