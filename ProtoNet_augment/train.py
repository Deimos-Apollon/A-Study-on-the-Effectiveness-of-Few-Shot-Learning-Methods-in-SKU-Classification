import logging
import mlflow
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import json

from tqdm import tqdm, trange
# import umap
from cycler import cycler
from PIL import Image
from torchvision import datasets, transforms

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)

IMG_RESIZE_SIZE = (112, 112)

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# reproducibility
random.seed(42)
np.random.seed(42) 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

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

    def get_model_accuracy(self, dataset):
        total_right = 0
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        for X, y in tqdm(data_loader):
            y_pred = self.predict(X)
            total_right += np.sum([y_p == y_t for y_p, y_t in zip(y_pred.cpu(), y.cpu())])
        acc = total_right / len(dataset)
        return acc

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_RESIZE_SIZE),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transforms = transforms.Compose([
    test_transforms
])


train_dir = None # путь к тренировочным данным, заполнить значением
test_dir = None # путь к тестовым данным, заполнить значением

train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
test_dataset = torchvision.datasets.ImageFolder(test_dir, test_transforms)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_data_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

num_classes = len(train_dataset.classes)

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

def load_supercls(MODEL_PATH, num_classes):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(parameters=weights)
    model.classifier[1] = nn.Linear(1280, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
    return model

def transform_trunk_supercls(trunk):
    trunk.classifier[1] = nn.Identity()
    return trunk

num_classes = len(train_dataset.classes)

MODEL_PATH = None # путь к тестовым данным, заполнить значением
trunk = load_supercls(MODEL_PATH, num_classes=num_classes).to(device)
trunk = transform_trunk_supercls(trunk)

# теперь можно брать embedder и cls по умолчанию из модели
model = MetricLearningClsModel(num_classes, trunk=trunk, embedder=nn.Identity(), EMBEDDING_SIZE=1280).to(device)
embedder = model.embedder
classifier = model.classifier

# Set optimizers
trunk_lr = 0.0001
# embedder_lr = 0.0001
classifier_lr = 0.001
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=trunk_lr, weight_decay=0.0001)
# embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=embedder_lr, weight_decay=0.0001)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr, weight_decay=0.0001)

# Set schedulers
trunk_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trunk_optimizer)
# embedder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(embedder_optimizer)
classifier_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer)

# Set the loss function
loss = losses.TripletMarginLoss(margin=0.1)
classification_loss = torch.nn.CrossEntropyLoss()

# Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=0.1)

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(
    train_dataset.targets, m=4, length_before_new_iter=len(train_dataset)
)

# Set other training parameters
batch_size = 16
num_epochs = 30

# Package the above stuff into dictionaries.
models = {"trunk": trunk, #"embedder": embedder, 
          "classifier": classifier}
optimizers = {
    "trunk_optimizer": trunk_optimizer,
    # "embedder_optimizer": embedder_optimizer,
    "classifier_optimizer": classifier_optimizer
}
loss_funcs = {"metric_loss": loss, "classifier_loss": classification_loss}
mining_funcs = {"tuple_miner": miner}
lr_schedulers = {"trunk_scheduler_by_plateau": trunk_scheduler,
                # "embedder_scheduler_by_plateau": embedder_scheduler,
                "classifier_scheduler_by_plateau": classifier_scheduler}

ROOT_DIR = None # путь к папке для выходных логов, заполнить значением

record_keeper, _, _ = logging_presets.get_record_keeper(
    f"{ROOT_DIR}/train_logs", 
    f"{ROOT_DIR}/train_tensorboard"
)
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": test_dataset}
model_folder = f"{ROOT_DIR}/train_saved_models"


# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(
    end_of_testing_hook=hooks.end_of_testing_hook,
    dataloader_num_workers=2,
    accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
)

end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester, dataset_dict, model_folder, test_interval=1, patience=4
)

trainer = trainers.TrainWithClassifier(
    models,
    optimizers,
    batch_size,
    loss_funcs,
    train_dataset,
    mining_funcs=mining_funcs,
    sampler=sampler,
    dataloader_num_workers=2,
    end_of_iteration_hook=hooks.end_of_iteration_hook,
    end_of_epoch_hook=end_of_epoch_hook,
    lr_schedulers=lr_schedulers
)

trainer.train(num_epochs=num_epochs)

class_to_idx_path = f'{ROOT_DIR}/class_to_idx.txt'
class_to_idx = train_dataset.class_to_idx

with open(class_to_idx_path, 'w') as fs:
    fs.writelines(f"{sku};{ind}\n" for sku, ind in class_to_idx.items())

model_save_path = f'{ROOT_DIR}/metric_learning_cls_model.pth'
torch.save({'num_classes': num_classes,
            'class_to_idx': class_to_idx,
            'state_dict': model.state_dict()}, model_save_path)

train_acc = model.get_model_accuracy(train_dataset)
val_acc = model.get_model_accuracy(test_dataset)
print("train acc", train_acc)
print("val acc", val_acc)
