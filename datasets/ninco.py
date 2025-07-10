import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = []
        for root, _, files in os.walk(img_dir):  
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.img_files.append(os.path.join(root, f))  
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_dataset(): 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                            std=[0.229, 0.224, 0.225]),
    ])

    img_dir = 'D:/non_crop/data/NINCO/NINCO_OOD_classes'
    return CustomImageDataset(img_dir=img_dir, transform=transform)
