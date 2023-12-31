import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import rle_decode 

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False, args=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            return image, mask

    # Perform one hot encoding on label
    def one_hot_encode(self, label, label_values):
        """
        Convert a segmentation image label array to one-hot format
        by replacing each pixel value with a vector of length num_classes
        # Arguments
            label: The 2D array segmentation image label
            label_values

        # Returns
            A 2D array with the same width and hieght as the input, but
            with a depth size of num_classes
        """
        semantic_map = []
        for colour in label_values:
            equality = np.equal(label, colour)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1)

        return semantic_map