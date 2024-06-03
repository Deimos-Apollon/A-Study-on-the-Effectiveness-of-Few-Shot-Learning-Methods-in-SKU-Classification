import os
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils import data
import pandas as pd
import torch, sys, os, pdb
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from random import random, randint

src_dir = None # директория с исходными данными, заполнить значением
dst_dir = None # директория для выходных данных, заполнить значением 

IMG_RESIZE_SIZE = (112, 112)


class AugmentationManager:
  def __init__(self, params={"input_size": (360, 120),
                             "rotate_degrees": 3,
                             "resize_min_size": (120, 40),
                             "resize_max_size": (240, 80),
                             "brightness": 0.1,
                             "contrast": 0.1,
                             "saturation": 0.1,
                             "blur_kernel_size": (5, 9),
                             "crop_size": (300, 100),
                             "distortion_scale": 0.5},
               probs={'rotate': 0.9,
                      'resize': 0.5,
                      'jitter': 0.9,
                      'blur': 0.8,
                      'crop': 0.6,
                      'perspective': 0.5}):
    self.rotate = transforms.RandomRotation(params['rotate_degrees'])
    def random_resize(img):
      new_img = torch.clone(img)
      rand_size = (randint(params["resize_min_size"][0], params["resize_max_size"][0]),
                   randint(params["resize_min_size"][1], params["resize_max_size"][1]))
      new_img = transforms.functional.resize(new_img, rand_size)
      new_img = transforms.functional.resize(new_img, params["input_size"])
      return new_img
    self.resize = random_resize
    self.jitter = transforms.ColorJitter(params["brightness"],
                                         params["contrast"],
                                         params["saturation"])
    self.blur = transforms.GaussianBlur(params["blur_kernel_size"])
    self.crop = transforms.RandomCrop(params["crop_size"])
    self.perspective = transforms.RandomPerspective(distortion_scale=params["distortion_scale"],
                                                    p=probs['perspective'],
                                                    fill=0)

    self.probs = probs

  def aug(self, img: torch.tensor):
      new_img = torch.clone(img)
      if random() <= self.probs['rotate']:
        new_img = self.rotate(new_img)
      if random() <= self.probs['resize']:
        new_img = self.resize(new_img)
      if random() <= self.probs['jitter']:
        new_img = self.jitter(new_img)
      if random() <= self.probs['blur']:
        new_img = self.blur(new_img)
      new_img = self.perspective(new_img) # probability is embedded in this transform
      if random() <= self.probs['crop']:
        new_img = self.crop(new_img)
      return new_img
  

params={"input_size": IMG_RESIZE_SIZE,
    "rotate_degrees": 8,
    "resize_min_size": (56, 56),
    "resize_max_size": (112, 112),
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "blur_kernel_size": (5, 9),
    "crop_size": (90, 90),
    "distortion_scale": 0.5}

probs={'rotate': 0.5,
        'resize': 0.3,
        'jitter': 0.3,
        'blur': 0.3,
        'crop': 0.3,
        'perspective': 0}

aug_manager = AugmentationManager(params, probs)

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


aug_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_RESIZE_SIZE),
    aug_manager.aug,
    transforms.Resize(IMG_RESIZE_SIZE)
])

from torch import imag


for classname in tqdm(os.listdir(src_dir)):
    classpath = os.path.join(src_dir, classname)
    dst_class_dir = os.path.join(dst_dir, classname)
    if not os.path.exists(dst_class_dir):
        os.mkdir(dst_class_dir)

    for filename in os.listdir(classpath):
        filepath = os.path.join(classpath, filename)
        img = Image.open(filepath)
        img = img.convert('RGB')
        img_tensor = val_transforms(img)
        for i in range(1, 5):
            dst_filename = filename.split('.')[0] + f'_aug{i}.jpg'
            dst_filepath =  os.path.join(dst_class_dir, dst_filename)

            new_img_tensor = aug_transforms(img)
            new_img = transforms.ToPILImage()(new_img_tensor)
            new_img.save(dst_filepath)
        dst_filepath =  os.path.join(dst_class_dir, filename)
        img.save(dst_filepath)