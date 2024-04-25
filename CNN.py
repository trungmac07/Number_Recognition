
import numpy as np # to handle matrix and data operation

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision


import PIL.Image as Image

import matplotlib.pyplot as plt




class CNN(nn.Module):
    
    def __init__(self, device = 'cpu'):
        super(CNN, self).__init__()
        
        self.device = device

        self.layers = nn.Sequential()

        self.layers.append(nn.Conv2d(1,32,kernel_size=5, padding=2, stride=1, device=self.device))
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(nn.LeakyReLU(0.01))
        self.layers.append(nn.Conv2d(32,64,kernel_size=5, padding=2, stride=1, device=self.device))
        self.layers.append(nn.Dropout(p=0.3))
        self.layers.append(nn.LeakyReLU(0.01))
        self.layers.append(nn.AvgPool2d(2))
        self.layers.append(nn.Conv2d(64,128,kernel_size=5, padding=2, stride=1, device=self.device))
        self.layers.append(nn.LeakyReLU(0.01))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(7*7*128, 512, device=self.device))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=0.3))
        self.layers.append(nn.Linear(512, 10, device=self.device))

        self.loss = nn.CrossEntropyLoss()
    
    def forward(self,x):
        return self.layers(x)

    def predict(self,x):
        img = torchvision.transforms.Resize((28,28))(x)
        
        if(len(img.size()) <= 4):
            img = img.unsqueeze(0)

        y_pred = F.softmax(self.forward(img))

        return torch.max(y_pred,1)[1]
    
    def probability_array(self,x):
        img = torchvision.transforms.Resize((28,28))(x)
        
        if(len(img.size()) <= 4):
            img = img.unsqueeze(0)

        y_pred = F.softmax(self.forward(img))

        return y_pred.squeeze().detach().numpy()
    

def fit(model, data, device = 'cpu'):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    EPOCHS = 10
    model.train()

    for e in range(EPOCHS):
        correct = 0
        for i, (x_batch,y_batch) in enumerate(data):
            x = Variable(x_batch).float().to(device)
            y = Variable(y_batch).to(device)
        
            optimizer.zero_grad()
            y_pred = model.forward(x)
            loss = model.loss(y_pred, y)
            loss.backward()
            optimizer.step()
            predicted = torch.max(y_pred,1)[1]
            correct += (predicted == y).sum()
            if i % 50 == 0:
                print("{:<15} {:<15} {:<30} {:<30}".format("Epoch: " + str(e), "| Batch: " + str(i), "| Loss: " + str(loss.item()), "| accuracy: " + str(float(correct/float(BATCH_SIZE*(i+1))))))
 


            

