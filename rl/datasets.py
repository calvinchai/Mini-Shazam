import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FMADatasetSpec(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.image_files = [f for f in os.listdir(directory)]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        # Convert image to tensor
        image = self.to_tensor(image)

        # Extract ID from filename
        image_id = os.path.splitext(self.image_files[idx])[0]

        return image, image_id
