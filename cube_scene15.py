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
rows = 8 #サンプリングマトリックスの行数
cols = 8 #サンプリングマトリックスの列数
M =  64#生成されるビデオのフレーム数
blk_size = cols
sampling_rate = M // (rows*cols)
cube_size = 384 // cols
# 平均と標準偏差を指定
mean = 0.0
std_dev = 1.0
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# sampling_matrix = torch.rand(M, rows, cols)
# matrix = torch.rand(M, rows, cols) #Generate the random matrix from 0 to 1
# sampling_matrix = torch.where(matrix > 0.3, torch.ones(M, rows, cols), torch.zeros(M, rows, cols))
# kernel = sampling_matrix.unsqueeze(1)
# kernel = torch.tensor(kernel).float().to(device)
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
model.load_state_dict(torch.load("/home/19x3039_kimishima/pytorch-classification-uncertainty/checkpoint/pretrain_scene15_acc_91.68_proposed_loss_resnet18.pth"))
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = 1e-3,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
transform_cube_train = transforms.Compose([
        transforms.RandomResizedCrop(cube_size, scale = (0.8, 0.8)),
        transforms.RandomHorizontalFlip(),
    ])
transform_cube_val = transforms.Compose([
        # transforms.CenterCrop(cube_size),
        transforms.RandomResizedCrop(cube_size, scale = (0.8, 0.8)),
        transforms.RandomHorizontalFlip()
    ])
# for inputs, targets in train_loader:
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
    correct_uncertainty = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets_ = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        reconstructed_cube_all = torch.randn(0, 3, cube_size, cube_size).to(device)
        y = torch.zeros(0, num_classes).to(device)
        targets = torch.zeros(0).int().to(device)
        cube_r = F.conv2d(inputs[:, 0:1, :, :], PhiWeightR, padding=0, stride=blk_size, bias=None)
        cube_g = F.conv2d(inputs[:, 1:2, :, :], PhiWeightG, padding=0, stride=blk_size, bias=None)
        cube_b = F.conv2d(inputs[:, 2:3, :, :], PhiWeightB, padding=0, stride=blk_size, bias=None)
        # img = inputs[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
        # plt.figure(figsize=(10, 4))
        # plt.subplot(131)
        # plt.imshow(img)
        # plt.savefig("/home/19x3039_kimishima/pytorch-classification-uncertainty/images/scene15_original.jpg")
        for i in range(M):
            max_r = torch.max(cube_r[:,i,:,:])
            min_r = torch.min(cube_r[:,i,:,:])
            normarized_cube_r = (cube_r[:,i,:,:] - min_r) / (max_r - min_r)
            max_b = torch.max(cube_b[:,i,:,:])
            min_b = torch.min(cube_b[:,i,:,:])
            normarized_cube_b = (cube_b[:,i,:,:] - min_b) / (max_b - min_b)
            max_g = torch.max(cube_g[:,i,:,:])
            min_g = torch.min(cube_g[:,i,:,:])
            normarized_cube_g = (cube_g[:,i,:,:] - min_g) / (max_g - min_g)
            reconstructed_cube = torch.stack([normarized_cube_r, normarized_cube_g, normarized_cube_b], dim = 1)
            reconstructed_cube_all = torch.cat([reconstructed_cube_all, reconstructed_cube], dim = 0)
            y_ = one_hot_embedding(targets_, num_classes)
            y_ = y_.to(device)
            y = torch.cat([y, y_], dim = 0)
            targets = torch.cat([targets, targets_], dim = 0)
        #     image = reconstructed_cube[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy() #ミニバッチに含まれる一つの画像を表示する
        #     plt.figure(figsize=(10, 4))
        #     plt.subplot(131)
        #     plt.imshow(image)
        #     plt.savefig("/home/19x3039_kimishima/pytorch-classification-uncertainty/images/scene15_cube_"+str(i)+"_.jpg")
        # exit()
        reconstructed_cube_all = transform_cube_train(reconstructed_cube_all)
        outputs_1, outputs_u_max, outputs = model(reconstructed_cube_all)
        #loss = ce_loss(targets, outputs, num_class, epoch, 10, device)
        #loss = edl_mse_loss(outputs, y.float(), epoch, num_class, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
        loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
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
        correct_uncertainty += mean_uncertainty.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    total_accuracy = correct / total
    total_uncertainty = correct_uncertainty / len(train_loader)
    return total_accuracy, total_uncertainty

        

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    loss_func_kd = KnowledgeDistillationLoss()
    total = 0
    num_class = 15
    correct_uncertainty = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets_ = inputs.to(device), targets.to(device)
            reconstructed_cube_all = torch.randn(0, 3, cube_size, cube_size).to(device)
            y = torch.zeros(0, num_classes).to(device)
            targets = torch.zeros(0).int().to(device)
            cube_r = F.conv2d(inputs[:, 0:1, :, :], PhiWeightR, padding=0, stride=blk_size, bias=None)
            cube_g = F.conv2d(inputs[:, 1:2, :, :], PhiWeightG, padding=0, stride=blk_size, bias=None)
            cube_b = F.conv2d(inputs[:, 2:3, :, :], PhiWeightB, padding=0, stride=blk_size, bias=None)
            for i in range(M):
                max_r = torch.max(cube_r[:,i,:,:])
                min_r = torch.min(cube_r[:,i,:,:])
                normarized_cube_r = (cube_r[:,i,:,:] - min_r) / (max_r - min_r)
                max_b = torch.max(cube_b[:,i,:,:])
                min_b = torch.min(cube_b[:,i,:,:])
                normarized_cube_b = (cube_b[:,i,:,:] - min_b) / (max_b - min_b)
                max_g = torch.max(cube_g[:,i,:,:])
                min_g = torch.min(cube_g[:,i,:,:])
                normarized_cube_g = (cube_g[:,i,:,:] - min_g) / (max_g - min_g)
                reconstructed_cube = torch.stack([normarized_cube_r, normarized_cube_g, normarized_cube_b], dim = 1)
                reconstructed_cube_all = torch.cat([reconstructed_cube_all, reconstructed_cube], dim = 0)
                y_ = one_hot_embedding(targets_, num_classes)
                y_ = y_.to(device)
                y = torch.cat([y, y_], dim = 0)
                targets = torch.cat([targets, targets_], dim = 0)
            reconstructed_cube_all = transform_cube_val(reconstructed_cube_all)
            outputs_1, outputs_u_max, outputs = model(reconstructed_cube_all)
            #loss = ce_loss(targets, outputs, num_class, epoch, 10, device)
            #loss = edl_mse_loss(outputs, y.float(), epoch, num_class, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
            loss = ce_loss(targets, outputs, num_class, epoch, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
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
            correct_uncertainty += mean_uncertainty.item()
            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    total_acc = correct/total
    total_uncertainty = correct_uncertainty / len(val_loader)
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(model.state_dict(), './checkpoint/scene15_blk_size_'+str(blk_size)+'_TTA_resnet18_ex.pth')
        best_acc = acc
    print("Best Accuracy:", best_acc)
    return total_acc, total_uncertainty
    
train_acc_list = []
train_uncertainty_list = []
test_acc_list = []
test__uncertainty_list = []
for epoch in range(total_epoch):
    train_acc, train_u = train(epoch+1)
    test_acc, test_u = test(epoch+1)
    train_acc_list.append(train_acc)
    train_uncertainty_list.append(train_u)
    test_acc_list.append(test_acc)
    test__uncertainty_list.append(test_u)
    scheduler.step()


# グラフの作成
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
x_value = list(range(1, len(train_acc_list)+1))
# Train Accuracyのプロット
axes[0].plot(x_value, train_acc_list, marker = "o", label='Train Accuracy')
axes[0].plot(x_value, test_acc_list, marker = "x", label='Test Accuracy')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Validation Accuracyのプロット
axes[1].plot(x_value, train_uncertainty_list, marker = "o", label='Train Uncertainty')
axes[1].plot(x_value, test__uncertainty_list, marker = "x", label='Test Uncertainty')
axes[1].set_title('Uncertainty')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Uncertainty')
axes[1].legend()

# 画像の保存と表示
plt.tight_layout()
plt.savefig('/home/19x3039_kimishima/pytorch-classification-uncertainty/images/scene15_acc_uncertainty_TTA_ex.jpg')
plt.show()
# def train_model(model, dataloader, num_classes, loss_func, optimizer, scheduler, num_epochs, device):
#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     loss_func_kd = KnowledgeDistillationLoss()
#     losses = {"loss": [], "phase": [], "epoch": []}
#     accuracy = {"accuracy": [], "phase": [], "epoch": []}
#     evidences = {"evidence": [], "type": [], "epoch": []}
#     transform_cube_train = transforms.Compose([
#         # transforms.CenterCrop(96),
#         transforms.RandomResizedCrop(cube_size, scale = (0.8, 0.8)),
#         transforms.RandomHorizontalFlip(),
#         # transforms.RandomVerticalFlip(),
#         # transforms.RandomVerticalFlip(p=0.5),
#         # transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.2)
#     ])
#     transform_cube_val = transforms.Compose([
#         transforms.CenterCrop(cube_size),
#         transforms.RandomResizedCrop(cube_size, scale = (0.8, 0.8)),
#         transforms.RandomHorizontalFlip(),
#     ])
#     accuracy_file = open("/home/19x3039_kimishima/pytorch-classification-uncertainty/epoch_uncertainty_scene15_blk_size_"+str(blk_size)+"_SR_"+str(sampling_rate)+".txt", "w")
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
#             for j, (inputs, targets) in enumerate(dataloaders[phase]):
#                 inputs = inputs.to(device)
#                 targets_ = targets.to(device)
#             # zero the parameter gradients
#                 optimizer.zero_grad()
#                 with torch.set_grad_enabled(phase == "train"):
#                     reconstructed_cube_all = torch.randn(0, 3, cube_size, cube_size).to(device)
#                     y = torch.zeros(0, num_classes).to(device)
#                     targets = torch.zeros(0).int().to(device)
#                     cube_r = F.conv2d(inputs[:, 0:1, :, :], PhiWeightR, padding=0, stride=blk_size, bias=None)
#                     cube_g = F.conv2d(inputs[:, 1:2, :, :], PhiWeightG, padding=0, stride=blk_size, bias=None)
#                     cube_b = F.conv2d(inputs[:, 2:3, :, :], PhiWeightB, padding=0, stride=blk_size, bias=None)
#                     for i in range(M):
#                         max_r = torch.max(cube_r[:,i,:,:])
#                         min_r = torch.min(cube_r[:,i,:,:])
#                         normarized_cube_r = (cube_r[:,i,:,:] - min_r) / (max_r - min_r)
#                         max_b = torch.max(cube_b[:,i,:,:])
#                         min_b = torch.min(cube_b[:,i,:,:])
#                         normarized_cube_b = (cube_b[:,i,:,:] - min_b) / (max_b - min_b)
#                         max_g = torch.max(cube_g[:,i,:,:])
#                         min_g = torch.min(cube_g[:,i,:,:])
#                         normarized_cube_g = (cube_g[:,i,:,:] - min_g) / (max_g - min_g)
#                         reconstructed_cube = torch.stack([normarized_cube_r, normarized_cube_g, normarized_cube_b], dim = 1)
#                         reconstructed_cube_all = torch.cat([reconstructed_cube_all, reconstructed_cube], dim = 0)
#                         y_ = one_hot_embedding(targets_, num_classes)
#                         y_ = y_.to(device)
#                         y = torch.cat([y, y_], dim = 0)
#                         targets = torch.cat([targets, targets_], dim = 0)
#                     if phase == "train":
#                         reconstructed_cube_all = transform_cube_train(reconstructed_cube_all)
#                     else:
#                         reconstructed_cube_all = transform_cube_val(reconstructed_cube_all)
#                     outputs_1, outputs_u_max, outputs = model(reconstructed_cube_all)
#                     # img = reconstructed_cube_all[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
#                     # plt.figure(figsize=(10, 4))
#                     # plt.subplot(131)
#                     # plt.imshow(img)
#                     # plt.savefig("/home/19x3039_kimishima/pytorch-classification-uncertainty/images/scene15.jpg")
#                     # exit()
                    
#                     # outputs = model(reconstructed_cube_all)
#                     _, preds = torch.max(outputs, 1)
#                     match = torch.reshape(torch.eq(preds, targets).float(), (-1, 1))
#                     acc = torch.mean(match)
#                     evidence = relu_evidence(outputs)
#                     param = torch.sum(evidence).item()
#                     alpha = evidence + 1
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
#                     # loss = proposed_ce_loss(targets, outputs, num_classes, epoch, 10, y.float(), 5, device)
#                     #loss = loss_func(targets, outputs, num_classes, epoch, 10, device)
#                     loss = loss_func(targets, outputs, num_classes, epoch, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
#                     # loss = proposed_ce_loss(targets, outputs, num_classes, epoch, 10, y.float(), 3
#                     #                         , device) + loss_func_kd(outputs_u_max, y.float(), outputs)
#                     # loss = ce_loss(targets, outputs, num_classes, epoch, 10, device) + proposed_kd_loss(outputs_1, y, param=3) + loss_func_kd(outputs_u_max, y.float(), outputs)
#                     # loss = loss_func(targets, outputs, num_classes, epoch, 10, device) + loss_func_kd(outputs_u_max, y.float(), outputs)
#                     # loss = loglikelihood_loss(targets, alpha, num_classes, epoch, 10, device)
#                     if phase == "train":
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == targets.data)
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
#                 path = "/home/19x3039_kimishima/pytorch-classification-uncertainty/results/" + str(dataset) +"_ResNet_18_SR_"+str(sampling_rate)+"_M_"+str(M)+"_.pth"
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

# epoch = 200
# num_classes = 15
# model = ResNet18(num_classes = num_classes, input_channels = 3)
# # model = ViT('B_16_imagenet1k', image_size=96, num_classes=15, in_channels = 3, pretrained=True)
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
# # model.load_state_dict(torch.load("/home/19x3039_kimishima/pytorch-cifar/checkpoint/pretrain_scene15_ce_new_proposed_kd_384_resnet18.pth"))
# model = model.to(device)
# optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum = 0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200)#T_max=20
# #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5, last_epoch=-1)
# loss_func = ce_loss
# model, metrics = train_model(model, dataloaders, num_classes, loss_func, optimizer, scheduler = scheduler, num_epochs = epoch, device = device)
# end_time = time.time()
# total_time = end_time - start_time
# print("Total Time:", total_time)