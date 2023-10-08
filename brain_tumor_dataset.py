from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

data_labels = {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}
label_to_int = {'glioma': 0, 'meningioma': 1, 'no_tumor': 2, 'pituitary': 3}

class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, labels, img_size=(224, 224), is_training=False):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.is_training = is_training
        self.transforms_dict = self._get_transforms()

    def _get_transforms(self):
        transforms_dict = {}
        
        common_transforms = [
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transforms_dict['base'] = transforms.Compose(common_transforms)
        
        if self.is_training:
            # Rotation
            for angle in [45, 90, 120, 180, 270, 300, 330]:
                transforms_dict[f'rot_{angle}'] = transforms.RandomRotation(angle)
            
            # Random Horizontal and Vertical Flip
            transforms_dict['hflip'] = transforms.RandomHorizontalFlip(p=0.5)
            transforms_dict['vflip'] = transforms.RandomVerticalFlip(p=0.5)
            
            # Random Brightness and Contrast
            transforms_dict['brightness_contrast'] = transforms.ColorJitter(brightness=0.4, contrast=0.6)
            
            # Random Affine
            transforms_dict['affine'] = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        
        return transforms_dict


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Convert label to integer
        label = label_to_int[label]
        
        image = Image.open(image_path).convert('RGB')
        
        image = self.transforms_dict['base'](image)

        if self.is_training:
            angle_key = random.choice(list(self.transforms_dict.keys())[1:]) 
            image = self.transforms_dict[angle_key](image)

        return image, label