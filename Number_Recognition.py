import tkinter
import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import torchvision.transforms as transforms

import base64
import io

import PIL.Image as Image 
import matplotlib.pyplot as plt

from CNN import * 
from Drawing_Board import *

model_path = "./model/mnist-cnn.pth"

cnn = CNN()
cnn.load_state_dict(torch.load(model_path))
cnn.eval()


if __name__ == '__main__':
    root = Tk()
    GUI(root,cnn)
    root.title('Number Recognition')
    root.mainloop()

