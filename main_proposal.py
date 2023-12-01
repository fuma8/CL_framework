import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
from torch.utils.data import DataLoader
from helpers import get_device, rotate_img, one_hot_embedding
import pickle
from losses import *
from vit import ViT
import PIL
import torchvision.models as models
from model import *

class CopiedCIFAR10Dataset():
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]


        return image, label

# with open('/home/19x3039_kimishima/pytorch-classification-uncertainty/data/train_cifar10_cube_0.25_4_bic.pkl', 'rb') as f:
#     trainloader = pickle.load(f)
# with open('/home/19x3039_kimishima/pytorch-classification-uncertainty/data/test_cifar10_cube_0.25_4_bic.pkl', 'rb') as f:
#     testloader = pickle.load(f)
# データの前処理を定義
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# # CIFAR-10データセットを読み込む
trainloader = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testloader = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# # DataLoaderを作成
train_loader = torch.utils.data.DataLoader(trainloader, batch_size=128, shuffle=True, num_workers = 2)
val_loader = DataLoader(testloader, batch_size=128, num_workers = 2)
dataloaders = {
    "train": train_loader,
    "val": val_loader,
}
def train_model(model, dataloader, num_classes, loss_func, optimizer, scheduler, num_epochs, device):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_func_kd = KnowledgeDistillationLoss()
    losses = {"loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}
    evidences = {"evidence": [], "type": [], "epoch": []}
    arr_dict = {}
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                print("Training...")
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            running_uncertainty = 0.0
            correct = 0
            cnt = 0
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
            # zero the parameter gradients
                optimizer.zero_grad()
                cnt += 1
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    y = one_hot_embedding(labels, num_classes)
                    y = y.to(device)
                    # outputs = model(inputs)
                    outputs_u_max, outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    if epoch < -1:
                        loss = loss_func(outputs, y.float(), epoch, num_classes, 10, device)
                    else:
                        loss = loss_func(outputs, y.float(), epoch, num_classes, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
                        # loss = loss_func(outputs, y.float(), epoch, num_classes, 10, device)
                        # loss = loss_func_kd(outputs_u_max, y.float(), outputs)
                    match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                    acc = torch.mean(match)
                    evidence = relu_evidence(outputs)
                    alpha = evidence + 1
                    uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                    mean_uncertainty = torch.mean(uncertainty)
                    total_evidence = torch.sum(evidence, 1, keepdim=True)
                    mean_evidence = torch.mean(total_evidence)
                    mean_evidence_succ = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * match
                    ) / torch.sum(match + 1e-20)
                    mean_evidence_fail = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * (1 - match)
                    ) / (torch.sum(torch.abs(1 - match)) + 1e-20)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_uncertainty += mean_uncertainty
            if scheduler is not None:
                if phase == "train":
                    scheduler.step()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_uncertainty = running_uncertainty / cnt

            losses["loss"].append(epoch_loss)
            losses["phase"].append(phase)
            losses["epoch"].append(epoch)
            accuracy["accuracy"].append(epoch_acc.item())
            accuracy["epoch"].append(epoch)
            accuracy["phase"].append(phase)

            print(
                "{} loss: {:.4f} acc: {:.4f}".format(
                    phase.capitalize(), epoch_loss, epoch_acc
                )
            )
            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "/home/19x3039_kimishima/pytorch-classification-uncertainty/results/pretrained_model.pth")


    time_elapsed = time.time() - since
    print(
    "Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60
    )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)

    return model, metrics

epoch = 1000
num_classes = 10
# model = ResNet18_pretrained()
model = ResNet18()
# model = models.resnet18(pretrained=True) 
# model.fc = nn.Linear(model.fc.in_features, num_classes)
# model = ViT("B_32_imagenet1k", pretrained=False, image_size=(32, 32), num_classes=10)
# model = ViT(
#     image_size = 32,
#     patch_size = 4,
#     num_classes = 10,
#     dim = 512,
#     depth = 6,
#     heads = 8,
#     mlp_dim = 512,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
device = get_device()
model = model.to(device)
# save_state_dict = torch.load("/home/19x3039_kimishima/pytorch-classification-uncertainty/results/pretrain_cifar_original_ce_95.pth")
# new_state_dict = {}
# for key, item in save_state_dict.items():
#     new_state_dict[key.replace("module.", "")] = item
# model.load_state_dict(new_state_dict)
# optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)#weight_decay=0.0005
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200)#T_max=20
loss_func = edl_mse_loss
#ResNet18でモデルを学習する

model, metrics = train_model(model, dataloaders, num_classes, loss_func, optimizer, scheduler = exp_lr_scheduler, num_epochs = epoch, device = device)