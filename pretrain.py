import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pickle
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from cube_conversion import *
from losses import *
from helpers import get_device, rotate_img, one_hot_embedding
import copy
from model import *
from pytorch_pretrained_vit import ViT
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from utils import progress_bar

start_time = time.time()
M = 1 #生成されるビデオのフレーム数
rows = 4 #サンプリングマトリックスの行数
cols = 4 #サンプリングマトリックスの列数
blk_size = cols
sampling_rate = M // (rows*cols)
# 平均と標準偏差を指定
mean = 0.0
std_dev = 1.0
batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PhiR = torch.rand(M, rows, cols)
PhiB = torch.rand(M, rows, cols)
PhiG = torch.rand(M, rows, cols)
PhiR = torch.where(PhiR > 0.3, torch.ones(M, rows, cols), torch.zeros(M, rows, cols))
PhiB = torch.where(PhiR > 0.3, torch.ones(M, rows, cols), torch.zeros(M, rows, cols))
PhiG = torch.where(PhiR > 0.3, torch.ones(M, rows, cols), torch.zeros(M, rows, cols))
PhiWeightR = PhiR.unsqueeze(1).to(device)
PhiWeightB = PhiR.unsqueeze(1).to(device)
PhiWeightG = PhiR.unsqueeze(1).to(device)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.ImageFolder(root="/home/19x3039_kimishima/pytorch-classification-uncertainty/data/15-Scene", transform = transform)

dataset_size = len(dataset)
train_ratio = 0.7
train_size = int(train_ratio * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size)
dataloaders = {
    "train": train_loader,
    "val": val_loader,
}
best_acc = 0
total_epoch = 200
num_classes = 15
model = ResNet18(num_classes, input_channels = 3)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = 1e-3,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# for inputs, labels in train_loader:
#     img = inputs[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
#     plt.figure(figsize=(10, 4))
#     plt.subplot(131)
#     plt.imshow(img)
#     plt.savefig("/home/19x3039_kimishima/pytorch-classification-uncertainty/images/scene15.jpg")
#     exit()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    loss_func_kd = KnowledgeDistillationLoss()
    train_loss = 0
    correct = 0
    total = 0
    num_class = 15
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        y = one_hot_embedding(targets, num_class)
        y = y.to(device)
        optimizer.zero_grad()
        outputs_1, outputs_u_max, outputs = model(inputs)
        loss = ce_loss(targets, outputs, num_class, epoch, 10, device)
        # loss = edl_mse_loss(outputs, y.float(), epoch, num_class, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
        #loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
        # loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + proposed_kd_loss(outputs_u_max, y, param=3)
        # loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + proposed_kd_loss(relu_evidence(outputs), y, param=3) + loss_func_kd(outputs_u_max, y.float(), outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        _, preds = torch.max(outputs, 1)
        match = torch.reshape(torch.eq(preds, targets).float(), (-1, 1))
        acc = torch.mean(match)
        evidence = relu_evidence(outputs)
        alpha = evidence + 1
        uncertainty = num_class / torch.sum(alpha, dim=1, keepdim=True)
        mean_uncertainty = torch.mean(uncertainty)
        mean_evidence_succ = torch.sum(
            torch.sum(evidence, 1, keepdim=True) * match
        ) / torch.sum(match + 1e-20)
        mean_evidence_fail = torch.sum(
            torch.sum(evidence, 1, keepdim=True) * (1 - match)
        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)
        accuracy_file.write(f"Train: Epoch {epoch}: Uncertainty {mean_uncertainty:.3f} Evidence_succ {mean_evidence_succ:.3f} Evidence_fail {mean_evidence_fail:.3f}\n")
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        accuracy_file.write(f"Train: Epoch {epoch}: Accuracy {acc:.3f}\n")

        

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    loss_func_kd = KnowledgeDistillationLoss()
    total = 0
    num_class = 15
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            y = one_hot_embedding(targets, num_class)
            y = y.to(device)
            outputs_1, outputs_u_max, outputs = model(inputs)
            loss = ce_loss(targets, outputs, num_class, epoch, 10, device)
            # loss = edl_mse_loss(outputs, y.float(), epoch, num_class, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
            #loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
            # loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + proposed_kd_loss(outputs_u_max, y, param=3)
            #loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + proposed_kd_loss(relu_evidence(outputs), y, param=3) + loss_func_kd(outputs_u_max, y.float(), outputs)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            _, preds = torch.max(outputs, 1)
            match = torch.reshape(torch.eq(preds, targets).float(), (-1, 1))
            acc = torch.mean(match)
            evidence = relu_evidence(outputs)
            alpha = evidence + 1
            uncertainty = num_class / torch.sum(alpha, dim=1, keepdim=True)
            mean_uncertainty = torch.mean(uncertainty)
            mean_evidence_succ = torch.sum(
            torch.sum(evidence, 1, keepdim=True) * match
            ) / torch.sum(match + 1e-20)
            mean_evidence_fail = torch.sum(
                torch.sum(evidence, 1, keepdim=True) * (1 - match)
            ) / (torch.sum(torch.abs(1 - match)) + 1e-20)
            accuracy_file.write(f"Val: Epoch {epoch}: Uncertainty {mean_uncertainty:.3f} Evidence_succ {mean_evidence_succ:.3f} Evidence_fail {mean_evidence_fail:.3f}\n")
            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    accuracy_file.write(f"Val: Epoch {epoch}: Accuracy {acc:.3f}\n")
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(model.state_dict(), './checkpoint/pretrain_scene15_ce_new_proposed_kd_384_resnet18_original_loss.pth')
        best_acc = acc
    print("Best Accuracy:", best_acc)
    
accuracy_file = open("/home/19x3039_kimishima/pytorch-classification-uncertainty/epoch_uncertainty_original_loss.txt", "w")
for epoch in range(total_epoch):
    train(epoch+1)
    test(epoch+1)
    scheduler.step()
# def train_model(model, dataloaders, num_classes, loss_func, optimizer, scheduler, num_epochs, device):
#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     loss_func_kd = KnowledgeDistillationLoss()
#     losses = {"loss": [], "phase": [], "epoch": []}
#     accuracy = {"accuracy": [], "phase": [], "epoch": []}
#     evidences = {"evidence": [], "type": [], "epoch": []}
#     accuracy_file = open("/home/19x3039_kimishima/pytorch-classification-uncertainty/epoch_uncertainty_outputs_kd.txt", "w")
#     for epoch in range(num_epochs):
#         print("Epoch {}/{}".format(epoch+1, num_epochs))
#         print("-" * 10)
#         for phase in ["train", "val"]:
#             if phase == "train":
#                 print("Training...")
#                 model.train()  
#             else:
#                 print("Validating...")
#                 model.eval() 

#             running_loss = 0.0
#             running_corrects = 0.0
#             correct = 0
#             # Iterate over data.
#             for j, (inputs, labels) in enumerate(dataloaders[phase]):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#             # zero the parameter gradients
#                 optimizer.zero_grad()
#                 with torch.set_grad_enabled(phase == "train"):
#                     y = one_hot_embedding(labels, num_classes)
#                     y = y.to(device)
#                     outputs_1, outputs_u_max, outputs = model(inputs)
#                     # img = reconstructed_cube_all[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
#                     # plt.figure(figsize=(10, 4))
#                     # plt.subplot(131)
#                     # plt.imshow(img)
#                     # plt.savefig("/home/19x3039_kimishima/pytorch-classification-uncertainty/images/scene15.jpg")
#                     # exit()
                    
#                     # outputs = model(reconstructed_cube_all)
#                     _, preds = torch.max(outputs, 1)
#                     match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
#                     acc = torch.mean(match)
#                     evidence = relu_evidence(outputs)
#                     alpha = evidence + 1
#                     enhanced_evidence = torch.zeros(inputs.shape[0], num_classes).to(device)
#                     for i in range(inputs.shape[0]):
#                         evidence_sum = torch.sum(evidence[i])
#                         enhanced_evidence[i][torch.nonzero(y[i]).item()] = evidence_sum
#                     uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
#                     mean_uncertainty = torch.mean(uncertainty)
#                     total_evidence = torch.sum(evidence, 1, keepdim=True)
#                     mean_evidence = torch.mean(total_evidence)
#                     mean_evidence_succ = torch.sum(
#                         torch.sum(evidence, 1, keepdim=True) * match
#                     ) / torch.sum(match + 1e-20)
#                     mean_evidence_fail = torch.sum(
#                         torch.sum(evidence, 1, keepdim=True) * (1 - match)
#                     ) / (torch.sum(torch.abs(1 - match)) + 1e-20)
#                     accuracy_file.write(f"Phase {phase}: Epoch {epoch + 1}: Uncertainty {mean_uncertainty:.3f} Evidence_succ {mean_evidence_succ:.3f} Evidence_fail {mean_evidence_fail:.3f}\n")
#                     #loss = loss_func(labels, outputs, num_classes, epoch, 10, device)
#                     #loss = loss_func(labels, outputs, num_classes, epoch, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
#                     #loss = loss_func(labels, outputs, num_classes, epoch, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs) + loss_func_kd(outputs_1, y.float(), outputs)
#                     loss = loss_func(labels, outputs, num_classes, epoch, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs) + proposed_kd_loss(outputs, y.float(), param = 3)
#                     if phase == "train":
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#             if scheduler is not None:
#                 if phase == "train":
#                     scheduler.step()
#             epoch_loss = running_loss / (len(dataloaders[phase].dataset) * M)
#             epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset) * M)
#             losses["loss"].append(epoch_loss)
#             losses["phase"].append(phase)
#             losses["epoch"].append(epoch)
#             accuracy["accuracy"].append(epoch_acc.item())
#             accuracy["epoch"].append(epoch)
#             accuracy["phase"].append(phase)
#             accuracy_file.write(f"Phase {phase}: Epoch {epoch + 1}: Accuracy {epoch_acc:.3f}\n")
#             print(
#                 "{} loss: {:.4f} acc: {:.4f}".format(
#                     phase.capitalize(), epoch_loss, epoch_acc
#                 )
#             )
#             # deep copy the model
#             if phase == "val" and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#                 dataset = "scene15"
#                 path = "/home/19x3039_kimishima/pytorch-classification-uncertainty/results/pretrain_" + str(dataset) +"_ResNet_18_SR.pth"
#                 torch.save(model.state_dict(), path)
#             if phase == "val":
#                 print("Best Acc:", best_acc)


#     time_elapsed = time.time() - since
#     print(
#     "Training complete in {:.0f}m {:.0f}s".format(
#         time_elapsed // 60, time_elapsed % 60
#     )
#     )
#     print("Best val Acc: {:4f}".format(best_acc))

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     metrics = (losses, accuracy)

#     return model, metrics


# model = ViT('B_16_imagenet1k', image_size=96, num_classes=15, in_channels = 3, pretrained=True)
# model.load_state_dict(torch.load("/home/19x3039_kimishima/pytorch-cifar/checkpoint/pretrain_scene15_ce_new_proposed_kd_384_resnet18.pth"))
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5, last_epoch=-1)
# loss_func = ce_loss
# model, metrics = train_model(model, dataloaders, num_classes, loss_func, optimizer, scheduler = scheduler, num_epochs = epoch, device = device)
# end_time = time.time()
# total_time = end_time - start_time
# print("Total Time:", total_time)