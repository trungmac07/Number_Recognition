import torch
import torchvision.transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision.transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision.transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class IntegratedGradient:
    def __init__(self, model, img_size=(28,28), device = 'cpu'):
        self.device = device
        self.model = model
        self.img_size = img_size
    
    def find_base_line(self, classes = 10):
        
        distribution = torch.tensor([[0.1 for i in range(classes)]])
        for i in range(100):
            y = self.model.forward(self.base_line)
            loss = F.cross_entropy(y, distribution) + torch.mean(self.base_line)
            grad = torch.autograd.grad(loss, self.base_line, create_graph=True)[0].squeeze()
            self.base_line = self.base_line - 0.05* grad

        print("Baseline probability:",F.softmax(self.model.forward(self.base_line)))

    def explain(self, x, y = None, model = None, step = 50, auto_baseline = False,  baseline = None):
        
        img = x.unsqueeze(0)
       
        if(model==None):
            model = self.model

        if(y == None):
            y = torch.argmax(model.forward(img))

        if(auto_baseline == True):
            baseline = self.find_base_line(classes=10)
            print("Baseline probability:",F.softmax(model.forward(baseline)))

        
        
        saliency_map = torch.zeros((1,self.img_size[0],self.img_size[1]), device=self.device, requires_grad=True)
        if(baseline == None):
            baseline = torch.zeros((1,self.img_size[0],self.img_size[1]), device=self.device,  requires_grad=True)

        dif = img - baseline

        x_ig = torch.stack([baseline + a/step*dif for a in range(step+1)]).squeeze(1)
        #print(x_ig.shape)
        model.zero_grad()
        y_pred = F.softmax(model.forward(x_ig))
        y_pred = torch.index_select(y_pred,1,y)
        grad = torch.autograd.grad(y_pred, x_ig, create_graph=True, grad_outputs=torch.ones_like(y_pred))[0]
 
        saliency_map = saliency_map + torch.sum(grad, dim = 0)

        saliency_map = 1.0/(step+1) * saliency_map * dif
        saliency_map = torch.sum(saliency_map, dim = 1)

        saliency_map = F.relu(saliency_map)

        min_val = torch.min(saliency_map)
        max_val = torch.max(saliency_map)

        
        saliency_map = 1/(max_val - min_val) * (saliency_map - min_val)

        return saliency_map
       






# class IntegratedGradient:
#     def __init__(self, model, img_size=(28,28), find_baseline=True):
#         self.model = model
#         self.img_size = img_size
#         self.base_line = torch.zeros((1,img_size[0],img_size[1])).unsqueeze(0).requires_grad_(True)
        
#         # if(find_baseline):
#         #     self.find_base_line()
    
#     # def find_base_line(self):
        
#     #     distribution = torch.tensor([[0.1 for i in range(10)]])
#     #     for i in range(100):
#     #         y = self.model.forward(self.base_line)
#     #         loss = F.cross_entropy(y, distribution) + torch.mean(self.base_line)
#     #         grad = torch.autograd.grad(loss, self.base_line, create_graph=True)[0].squeeze()
#     #         self.base_line = self.base_line - 0.05* grad

#     #     print("Baseline probability:",F.softmax(self.model.forward(self.base_line)))


#     def explain(self, x, step = 100):
#         #return self.base_line.squeeze()
#         img = torchvision.transforms.Resize(self.img_size)(x)
#         if(len(img.size()) <= 3):
#             img = img.unsqueeze(0)
#         dif = img - self.base_line
#         saliency_map = torch.zeros(self.img_size)
#         saliency_map = saliency_map.unsqueeze(0)
#         saliency_map = saliency_map.unsqueeze(0)

#         for a in range(step+1):
#             x_ig = self.base_line + a/step*dif
#             self.model.zero_grad()
#             y_pred = torch.max(F.softmax(self.model.forward(x_ig)))
#             grad = torch.autograd.grad(y_pred, x_ig, create_graph=True)[0]
#             saliency_map += 1/(step+1) * grad

#         saliency_map *= dif  
            
#         min_val = torch.min(saliency_map)
#         max_val = torch.max(saliency_map)

#         saliency_map = 1/(max_val - min_val) * (saliency_map - min_val) 

#         return saliency_map.squeeze()
