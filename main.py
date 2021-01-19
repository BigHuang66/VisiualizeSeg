from __future__ import print_function, division
import torch
import os
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
from models.my_model import *
from train import *
from datasets.get_data import * 



# my_net
# model = Model()
# model = model.to(device)
# print(model)

# Resnet18

model_ft = models.alexnet(num_classes = 5, pretrained=False)
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 5)
model_ft = model_ft.to(device)


#alpha = np.array([0.006, 1, 0.42, 0.23, 0.17, 0.74, 0.55, 0.50], dtype=np.float32)

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model_ft.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
epochs = 100


train_model(model_ft, criterion, optimizer, scheduler, num_epochs=epochs, visdom_flag=False)
#model.load_state_dict(torch.load('./checkpoints/my_layer.pth'))


