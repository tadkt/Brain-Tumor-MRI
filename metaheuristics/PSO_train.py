import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt

from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data 
print('Preparing data...')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize()
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.ImageFolder(root='/Users/hanguyen/Documents/Prj_Brain_Tumor/Project-Brain-Tumor-MRI/data/Training', train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.ImageFolder(root='/Users/hanguyen/Documents/Prj_Brain_Tumor/Project-Brain-Tumor-MRI/data/Testing', train=False, transform=transform_train)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

classes = ('glioma','meningioma','notumor','pituitary')

models = []
for i in range(30):
    # models.append(VGG("VGG3").to(device))
    models.append(LeNet().to(device))
    # models[i].apply(weights_init)

criterion = nn.CrossEntropyLoss()
result = PSO_train(models, criterion, 10, Wmax=0.95, Wmin=0.25, c1=2, c2=2, trainloader=trainloader, testloader=testloader)


# acc-epoch
plt.figure()
plt.plot(result['train_acc_history'], label='train')
plt.plot(result['test_acc_history'], label='test')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('acc history')
plt.legend(loc='upper left')
plt.savefig("checkpoint_pso/epoch_acc.png")
plt.show()
