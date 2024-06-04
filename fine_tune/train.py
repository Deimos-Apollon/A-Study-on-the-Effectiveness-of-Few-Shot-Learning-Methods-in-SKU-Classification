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
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm
import copy
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

sns.set_theme()

# reproducibility
random.seed(42)
np.random.seed(42) 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

TRAIN_PATH = f"/net/freenas/QNAS/MOVED/SHELFMATCH/R&D/Emelyanov_Dmitry/diplom/rp2k_dataset_my_k_3/train"
TEST_PATH = f"/net/freenas/QNAS/MOVED/SHELFMATCH/R&D/Emelyanov_Dmitry/diplom/rp2k_dataset_my_k_10/test"

RES_DIR = "/home/emelyanov/shelfmatch/diplom/fine_tune"
RES_MODEL_DIR = f"{RES_DIR}/model"

val_accuracy = []
train_accuracy = []
val_loss = []
train_loss = []
epochs = []

device = torch.device('cuda:0')
LR = 0.001
N_EPOCH = 60
batch_size = 16

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

def train_model(model, loss, optimizer, scheduler, num_epochs):
    best_acc = 0
    best_epoch = 0
    for epoch in trange(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)
        gt = []
        net_outputs = []
        epochs.append(epoch + 1)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                    else:
                        gt.extend(labels.data.cpu().numpy())
                        net_outputs.extend(preds_class.data.cpu().numpy())

                # statistics
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            if phase == 'train':
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)
                train_accuracy.append(epoch_acc.item())
                train_loss.append(epoch_loss)
            else:
                bacc = balanced_accuracy_score(gt, net_outputs)
                print('{} Loss: {:.4f}, balanced_accuracy_score: {:.4f}, accuracy_score: {:.4f}'.format(phase, epoch_loss, bacc, accuracy_score(gt, net_outputs)), flush=True)
                val_accuracy.append(accuracy_score(gt, net_outputs).item())
                val_loss.append(epoch_loss)
                scheduler.step(epoch_loss)

            if phase == 'val' and bacc >= best_acc:
                best_acc = bacc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch+1

    model.load_state_dict(best_model_wts)
    return model, best_acc, best_epoch

weights = EfficientNet_V2_S_Weights.DEFAULT
net = efficientnet_v2_s(parameters=weights)
num_ftrs = net.classifier[1].in_features
net.classifier[1] = nn.Linear(num_ftrs, 100) #меняем количество слоев

train_dataset = torchvision.datasets.ImageFolder(TRAIN_PATH, train_transforms)
val_dataset = torchvision.datasets.ImageFolder(TEST_PATH, train_transforms)

NUM_CLASSES = len(train_dataset.classes)
class_to_idx = train_dataset.class_to_idx 
print(f"Num classes: {NUM_CLASSES}")


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

net.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES) #меняем количество слоев
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
net = net.to(device)

save_name = f'{RES_MODEL_DIR}/efficientnet_v2_s_from_sc_few_shot.pth.tar'
model, best_acc, best_epoch = train_model(net, loss, optimizer, scheduler, num_epochs=N_EPOCH)
print('best bacc', best_acc)
print('best acc', val_accuracy[best_epoch-1])
torch.save({'num_classes': NUM_CLASSES,
            'class_to_idx': class_to_idx,
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler,
            }, save_name)

# saving accuracy statistics
plt.plot(epochs, [x*100 for x in train_accuracy], label='Train accuracy')
plt.plot(epochs, [x*100 for x in val_accuracy], label='Validation accuracy')
plt.axvline(x=best_epoch, ls='--', color='red', label='model best state')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.savefig(f'{RES_DIR}/accuracy.png')
plt.close()

# saving loss statistics
plt.plot(epochs, train_loss, label = 'Train loss')
plt.plot(epochs, val_loss, label = 'Validation loss')
plt.axvline(x=best_epoch, ls='--', color='red', label='model best state')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.savefig(f'{RES_DIR}/loss.png')
plt.close()

