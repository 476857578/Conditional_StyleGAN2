from ast import Index
from tkinter import image_names
import numpy as np
import os
import torch
import random
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image
import cv2
import torchvision.transforms as transforms
    
class BubbleDataset(Dataset):
    def __init__(self, data_root, excel_dir):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.images_path = list(self.data_root.glob("*.png"))
        # self.images_path = random.sample(self.images_path, 200)
        self.bub_num = len(self.images_path)

        self.excel_dir = Path(excel_dir)
        if not self.excel_dir.exists():
            raise ValueError("Instance excel dir doesn't exists.")
        self.df = pd.read_csv(self.excel_dir, encoding='gbk')

    def __len__(self):
        # return 200
        return self.bub_num

    def __getitem__(self, index):
        path = self.images_path[index]
        name = self.df.iloc[index, 1]
        k1 = self.df.iloc[index, 2:].tolist()
        k1 = torch.tensor(k1, dtype=torch.float)

        image = Image.open(path)
        if not image.mode == "L":
            image = image.convert("L")
        image_tensor = transforms.ToTensor()(image)

        k2 = self.df.iloc[random.randint(0, self.bub_num - 1), 2:].tolist()
        k2 = torch.tensor(k2, dtype=torch.float)

        k3 = self.df.iloc[random.randint(0, self.bub_num - 1), 2:].tolist()
        k3 = torch.tensor(k3, dtype=torch.float)

        return image_tensor, k1, k2, k3