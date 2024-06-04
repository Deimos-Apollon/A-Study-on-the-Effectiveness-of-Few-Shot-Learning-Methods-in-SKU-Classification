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

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from time import time_ns
sns.set_theme()

MODEL_PATH = None # путь к тренировочным данным, заполнить значением 
TEST_PATH = None # путь к тренировочным данным, заполнить значением 

IMG_RESIZE_SIZE = (112, 112)

class MetricLearningClsModel(nn.Module):
    def __init__(self, num_classes, trunk=None, embedder=None, classifier=None, EMBEDDING_SIZE=128):
        super().__init__()
        
        # параметры по умолчанию
        if not trunk:
            weights = EfficientNet_V2_S_Weights.DEFAULT
            trunk = efficientnet_v2_s(parameters=weights)
            trunk.classifier[1] = nn.Identity()
        if not embedder:
            embedder = nn.Sequential(nn.Linear(in_features=1280, out_features=EMBEDDING_SIZE))
        if not classifier:
            classifier = nn.Sequential(nn.Linear(in_features=EMBEDDING_SIZE, out_features=num_classes))

        self.embedder = embedder
        self.trunk = trunk
        self.classifier = classifier

    def forward(self, X):
        X = self.trunk(X)
        X = self.embedder(X)
        X = self.classifier(X)
        return X
    
    def predict(self, X):
        self.eval()
        with torch.inference_mode():
            y_pred = []
            X = X.to(device)
            y_logits = self.forward(X)
            y_pred = torch.argmax(y_logits, dim=1)
            return y_pred

# reproducibility
random.seed(42)
np.random.seed(42) 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

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

MODEL_PATH = None # путь к обученной модели
checkpoint = torch.load(MODEL_PATH)
NUM_CLASSES = checkpoint['num_classes']
class_to_idx = checkpoint['class_to_idx'] 
print(f"Num classes: {NUM_CLASSES}")

net = MetricLearningClsModel(num_classes=NUM_CLASSES)
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
        y_preds = net.predict(X)
        total_time = time_ns() - start_time
        total_time /= 1e6
        times.append(total_time)
        y_trues_all.extend(y.tolist())
        y_preds_all.extend(y_preds.tolist())

bacc_val = balanced_accuracy_score(y_trues_all, y_preds_all)
f1 = f1_score(y_trues_all, y_preds_all, average='macro')
np.mean(times)
print(f"Сбалансированная точность: {bacc_val}")
print(f"F1-score: {f1}")
print(f"Среднее время: {np.mean(times)}")