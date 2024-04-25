import torch
import torchvision.transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
class IntergratedGradient:
    def __init__(self, model, img_size=(28,28), find_baseline=True):
        self.model = model
        self.img_size = img_size
        self.base_line = torch.zeros((1,img_size[0],img_size[1])).unsqueeze(0).requires_grad_(True)

        # if(find_baseline):
        #     self.find_base_line()
    
    # def find_base_line(self):
        
    #     distribution = torch.tensor([[0.1 for i in range(10)]])
    #     for i in range(100):
    #         y = self.model.forward(self.base_line)
    #         loss = F.cross_entropy(y, distribution) + torch.mean(self.base_line)
    #         grad = torch.autograd.grad(loss, self.base_line, create_graph=True)[0].squeeze()
    #         self.base_line = self.base_line - 0.05* grad

    #     print("Baseline probability:",F.softmax(self.model.forward(self.base_line)))


    def explain(self, x, step = 100):
        #return self.base_line.squeeze()
        img = torchvision.transforms.Resize(self.img_size)(x)
        if(len(img.size()) <= 3):
            img = img.unsqueeze(0)
        dif = img - self.base_line
        saliency_map = torch.zeros(self.img_size)
        saliency_map = saliency_map.unsqueeze(0)
        saliency_map = saliency_map.unsqueeze(0)

        for a in range(step+1):
            x_ig = self.base_line + a/step*dif
            self.model.zero_grad()
            y_pred = torch.max(F.softmax(self.model.forward(x_ig)))
            grad = torch.autograd.grad(y_pred, x_ig, create_graph=True)[0]
            saliency_map += 1/(step+1) * grad

        saliency_map *= dif  
            
        min_val = torch.min(saliency_map)
        max_val = torch.max(saliency_map)

        saliency_map = 1/(max_val - min_val) * (saliency_map - min_val) 

        return saliency_map.squeeze()
