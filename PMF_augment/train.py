import json
import os
import numpy as np
import time
import random
import torch
import torchvision.transforms
import torchvision.transforms as transforms
from tqdm import tqdm

from pmf_cvpr22.models import ProtoNet_Finetune
from pmf_cvpr22.models import get_model, get_backbone
from dotmap import DotMap
from PIL import Image

# os.environ['TERM'] = 'linux'
# os.environ['TERMINFO'] = '/etc/terminfo'

# reproducibility
random.seed(42)
np.random.seed(42) 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# args
args = DotMap()
args.deploy = 'finetune'
args.arch = 'dino_small_patch16'
args.no_pretrain = True
args.eval = True
args.resume = "MODEL_CHECKPOINT.pt"

args.eval = True
args.aug_prob = 0.5
args.ada_lr = 0.001
args.ada_steps = 1
# args.resume = r"C:\Users\Deimos\Desktop\Programming\shelfmatch_test_task\PMF\MODEL_CHECKPOINT.pt"

# args.ada_steps = 5    мне не хватает ОЗУ для этой задачи
# args.ada_lr = 0.001
# args.aug_prob = 0
# args.aug_types = ["color", ]

# model
device = 'cpu' if not torch.cuda.is_available() else 'cuda'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
model = get_model(args)
model.to(device)


# image transforms
def test_transform():
    def _convert_image_to_rgb(im):
        return im.convert('RGB')

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


preprocess = test_transform()


def inference(query, supp_x, labels, supp_y):
    '''
    query: PIL image
    supp_x: list of PIL images
    supp_y: list of labels per image
    labels: list of class names
    '''

    with torch.no_grad():
        # query image
        query = preprocess(query).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, H, W)

        supp_x = torch.stack(supp_x, dim=0).unsqueeze(0).to(device)  # (1, n_supp*n_labels, 3, H, W)
        supp_y = torch.tensor(supp_y).long().unsqueeze(0).to(device)  # (1, n_supp*n_labels)

        with torch.cuda.amp.autocast(True):
            output = model(supp_x, supp_y, query)  # (1, 1, n_labels)

        probs = output.softmax(dim=-1).detach().cpu().numpy()

        return {k: float(v) for k, v in zip(labels, probs[0, 0])}


def read_small_dataset_train(root):
    train_x = []
    train_y = []
    labels = os.listdir(f"{root}/train")

    for (curr_dir, _, filenames) in os.walk(f"{root}/train"):
        label = curr_dir.split('/')[-1]
        for filename in filenames:
            image_path = f"{curr_dir}/{filename}"
            image = Image.open(image_path)

            x = preprocess(image)
            y = labels.index(label)

            train_x.append(x)
            train_y.append(y)
    return train_x, train_y, labels


def evaluate_acc(supp_x, labels, supp_y, queries, true_labels, log_filename):
    correct = 0
    log_json = {}
    i = 0
    for idx, (true, query) in tqdm(
            enumerate(zip(true_labels, queries)),
            total=len(queries)):
        i += 1
        if i == 30:
            break
        ans = inference(query, supp_x, labels, supp_y)
        pred = max(ans, key=lambda x: ans[x])
        pred = labels.index(pred)
        if true == pred:
            correct += 1
        ans = {i: (el, ans[el]) for i, el in enumerate(ans)}
        log_json[idx] = {
            "filename": query.filename,
            "true": true,
            "pred": pred,
            "ans": ans
        }
    with open(log_filename, 'w') as file:
        json.dump(log_json, file, indent=4)
    return correct


root = "task2_splitted_80_20"
supp_x, supp_y, labels = read_small_dataset_train(root)

queries = []
true_labels = []

for (curr_dir, _, filenames) in os.walk(f"{root}/test"):
    label = curr_dir.split('/')[-1]
    for filename in filenames:
        image_path = f"{curr_dir}/{filename}"
        image = Image.open(image_path)
        x = image
        # x = preprocess(image) # не надо предобрабатывать
        y = labels.index(label)

        queries.append(x)
        true_labels.append(y)

query = queries[0]
query = preprocess(query).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, H, W)
supp_x = torch.stack(supp_x, dim=0).unsqueeze(0).to(device)  # (1, n_supp*n_labels, 3, H, W)
supp_y = torch.tensor(supp_y).long().unsqueeze(0).to(device)  # (1, n_supp*n_labels)

with torch.cuda.amp.autocast(True):
    output = model(supp_x, supp_y, query)  # (1, 1, n_labels)

probs = output.softmax(dim=-1).detach().cpu().numpy()
print(probs)

torch.save(model.state_dict(), "MODEL_CHECKPOINT.pt")
print("MODEL SAVED!")

log_filename = "log.json"
correct_num = evaluate_acc(supp_x, labels, supp_y, queries, true_labels, log_filename)
print(correct_num / len(queries))
