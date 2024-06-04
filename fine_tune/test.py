import os
import numpy as np
import random
from torchvision import datasets, models, transforms
from torch.utils import data
import pandas as pd
import json
import torch, sys, os, pdb
import torchvision
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm
import copy
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from sklearn.metrics import balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from time import time_ns

sns.set_theme()

# reproducibility
random.seed(42)
np.random.seed(42) 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

MODEL_PATH = None # путь к модели, заполнить значением
TEST_PATH = None # путь к тестовым данным, заполнить значением

device = torch.device('cuda:0')

train_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


val_dataset = torchvision.datasets.ImageFolder(TEST_PATH, train_transforms)

checkpoint = torch.load(MODEL_PATH)
NUM_CLASSES = checkpoint['num_classes']
class_to_idx = checkpoint['class_to_idx']
weights = EfficientNet_V2_S_Weights.DEFAULT
net = efficientnet_v2_s(parameters=weights)
num_ftrs = net.classifier[1].in_features
net.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES) #меняем количество слоев
net.load_state_dict(checkpoint['state_dict'])
net.to(device)

data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
net.eval()

idx_to_class = {v: k for k, v in class_to_idx.items()}
y_trues_all = []
y_preds_all = []
times = []
for X, y in tqdm(data_loader):
    with torch.inference_mode():
        X = X.to(device)
        start_time = time_ns()
        y_preds = net.forward(X).argmax(dim=1)
        total_time = time_ns() - start_time
        total_time /= 1e6
        times.append(total_time)

        y_trues_all.extend(y.tolist())
        y_preds_all.extend(y_preds.tolist())

bacc_val = balanced_accuracy_score(y_trues_all, y_preds_all)
f1 = f1_score(y_trues_all, y_preds_all, average='macro')
print(f"Сбалансированная точность: {bacc_val}")
print(f"F1-score: {f1}")
print(f"Среднее время: {np.mean(times)}")
