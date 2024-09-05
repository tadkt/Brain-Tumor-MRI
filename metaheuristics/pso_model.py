import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import os
import numpy as np
import copy
import time
import torchvision.transforms as transforms
import torch.optim as optim

class BrainTumorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16,32,5)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32,64,3)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64,128,3)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*5*5, 512) #fully connected layer 1
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3)
        x = torch.sigmoid(self.fc4)

        return x

def particle_swarm_optimization_train(models, criterion, epochs, Wmax, Wmin, c1, c2, train_loader, test_loader):
    '''
    :param models:
    :param epochs:
    :param Wmax:
    :param Wmin:
    :param c1:
    :param c2:
    :param trainloader:
    :param testloader:
    :return:
    '''
    num_models = len(models)

    # Initialize the velocity of each model
    max_V = torch.tensor(0.1).cuda
    current_V = {}
    current_position = {}
    best_position = {}
    current_acc = []
    best

