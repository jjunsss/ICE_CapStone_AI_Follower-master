import os
import glob
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np

from torchvision import datasets, transforms
import PIL.Image

def get_x(path, width):
    return (float(int(path.split("_")[1])) - width / 2) / (width / 2)

def get_y(path, height):
    return (float(int(path.split("_")[2])) - height / 2) / (height / 2)

def get_z(path):
    return float(int(path.split("_")[3]))

class XYDataset(torch.utils.data.Dataset):
    def __init__(self, dir, pre = True):
        self.dir = dir
        self.image_paths = glob.glob(os.path.join(self.dir, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
        self.preprocessing = pre

    def __len__(self):
        return len(self.image_paths)

    def getxyz(self, image_path, width, height):
        x = float(get_x(os.path.basename(image_path), width))
        y = float(get_y(os.path.basename(image_path), height))
        z = float(get_z(os.path.basename(image_path)))
        return x, y, z

    def preprocess(self, image, pre):
        if pre == True:
            image = self.color_jitter(image)
            image = transforms.functional.resize(image, (224, 224))
            image = transforms.functional.to_tensor(image)
            image = transforms.functional.gaussian_blur(image, 9)
            image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            image = transforms.functional.resize(image, (224, 224))
            image = transforms.functional.to_tensor(image)
            image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path)
        width, height = image.size

        x, y, z = self.getxyz(image_path, width, height)

        if float(np.random.rand(1)) > 0.5:
            image = transforms.functional.hflip(image)
            x = -x

        image = self.preprocess(image, self.preprocessing)

        return image, torch.tensor([x, y, z]).float()