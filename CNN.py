
import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision

import base64
import io

import PIL.Image as Image

import matplotlib.pyplot as plt




class CNN(nn.Module):
    
    def linear(self, x):
        return x
    
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_layers = nn.ModuleList([nn.Conv2d(1,32,5), 
                                     nn.Conv2d(32,32,5), 
                                     nn.MaxPool2d(2), 
                                     nn.Dropout(p=0.5), 
                                     nn.Conv2d(32,64,5), 
                                     nn.MaxPool2d(2), 
                                     nn.Dropout(p=0.5),  
                                     
                                     ])
        self.fc_layers = nn.ModuleList([nn.Linear(576,256),
                                        nn.Dropout(p=0.2), 
                                        nn.Linear(256,10),])  
        self.cnn_act = [F.relu,       
                        self.linear,         
                        F.relu,         
                        self.linear,       
                        F.relu,             
                        self.linear,     
                        self.linear,]       
        self.fc_act = [  F.relu,               
                            self.linear,                  
                            F.softmax]
        
        self.n_cnn = len(self.cnn_layers)
        self.n_fc = len(self.fc_layers)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self,x):
        X = torch.clone(x)
        for i in range(self.n_cnn):
            X = self.cnn_layers[i](X)
            X = self.cnn_act[i](X)
        X = X.view(-1, 576)
        for i in range(self.n_fc):
            X = self.fc_layers[i](X)
            X = self.fc_act[i](X)
            #print(X.shape)
        return X
    
    def predict(self,x):
            img = torchvision.transforms.Resize((28,28))(x)
            print(img.shape)
            y_pred = torch.max(self.forward(img),1)[1]
            return y_pred

    def ig(self,sample):
        x,y = sample
        y_pred = self.forward(x)
        #loss = self.loss(y_pred, y)
        print(y_pred.grad)
    
    
            

