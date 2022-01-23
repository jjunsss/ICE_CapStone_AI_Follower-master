import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import os
import glob
import PIL.Image
import numpy as np
import xyzoutput
import model

from torchvision import datasets, models

EPOCHS = 40
BATCH_SIZE = 8
DEVICE = torch.device('cuda')

check_pre = True
dataset1 = xyzoutput.XYDataset('C:/Users/JunsuKim/multitasking/dataset_xy', check_pre)

check_pre = False
dataset2 = xyzoutput.XYDataset('C:/Users/JunsuKim/multitasking/dataset_xy', check_pre)

dataset = dataset1 + dataset2

test_percent = 0.15
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

resnet = model.mix_model().to(DEVICE)

optimizer = optim.Adam(resnet.parameters(), lr=0.0001)
loss1 = nn.MSELoss()
loss2 = nn.BCELoss()
best_accuracy = 0.0
best_loss = 1.0
'''
for x, y in train_dataloader:
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    pos, chk = resnet(x)
    print(pos)
    print(chk)
    break
'''


def train(Epoch, model, data, optim, log_interval=8):
    model.train()

    for batch_idx, (images, labels) in enumerate(data):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optim.zero_grad()
        pos, chk = model(images)
        reg_loss = loss1(pos, labels[:, :2])
        cls_loss = loss2(chk, labels[:, 2].reshape(-1,1))
        print(pos[:,0].size())
        print(pos)
        break

        alpha = 0.1
        loss = reg_loss + alpha * cls_loss
        loss.backward()
        optim.step()

        if batch_idx % log_interval == 0:
            print("train combine_loss Epoch: {} [ {:.0f}% ] \t Train Loss : {:.6f} \t"
                  "reg_loss = {:.6f} \t cls_loss = {:.6f}".format(
                Epoch, 100. * batch_idx / len(data), loss.item(), reg_loss.item(),
                cls_loss.item()
            ))


def eval(Epoch, model, test_data,path, log_interval = 8, ):
    global best_accuracy, best_loss
    model.eval()
    test_loss = 0.0
    error_count = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_data):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            pos, chk = model(images)

            test_loss += float(loss1(pos, labels[:, :2]))
            error_count += torch.sum(torch.abs(torch.round(chk) - labels[:, 2].reshape(-1,1)))
        test_loss /= len(test_data)
        test_accuracy = (1.0 - (float(error_count) / len(test_dataset))) * 100.
        print('Z(Classification) Epoch: {} \t Acuuracy : {:.2f}%'.format(Epoch, test_accuracy))
        print("X,Y(regression) Epoch : {} \t Test_Loss : {:.6f}".format(Epoch, test_loss))
        if best_loss >= test_loss and (best_accuracy <= test_accuracy):
            torch.save(model.state_dict(), path)
            best_loss = test_loss
            best_accuracy = test_accuracy


best_model_path = "best_model_resnet18.pth"
for epochs in range(EPOCHS + 1):

    train(epochs, resnet, train_dataloader, optimizer, 8)
    #eval(epochs, resnet, test_dataloader, best_model_path, 8)