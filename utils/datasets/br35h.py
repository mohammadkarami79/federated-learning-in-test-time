import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging

class Br35HDataset(Dataset):
    """
    Dataset class for Br35H brain tumor dataset
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['no', 'yes']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []
        valid_exts = {'.jpg', '.jpeg', '.png'}
        logger = logging.getLogger(__name__)
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                ext = os.path.splitext(img_name)[1].lower()
                if ext in valid_exts:
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        if len(self.images) == 0:
            logger.warning(f"BR35H dataset appears empty at {self.root_dir}. Ensure 'no/' and 'yes/' contain images with extensions: {sorted(list(valid_exts))}")
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_br35h_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),  # Larger resize for better crops
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crops
            transforms.RandomHorizontalFlip(p=0.5),              # Horizontal flip
            transforms.RandomVerticalFlip(p=0.3),                # Vertical flip for medical
            transforms.RandomRotation(15),                       # Rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Color
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

def get_br35h_info():
    return {
        'num_classes': 2,
        'input_channels': 3,
        'input_size': 224,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    }
