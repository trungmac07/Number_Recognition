from tkinter import *
from tkinter import ttk, colorchooser
from PIL import ImageGrab

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GUI:

    def __init__(self,master,model):
        self.prediction = StringVar()
        self.prediction.set("Prediction")
        self.model = model
        self.img = None
        self.master = master
        self.color_fg = 'white'
        self.color_bg = 'black'
        self.old_x = None
        self.old_y = None
        self.penwidth = 27
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)

        self.old_x = e.x
        self.old_y = e.y

        

    def save(self):
        # Get Canvas Widget Coordinates
        x = self.c.winfo_rootx()
        y = self.c.winfo_rooty()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        
        # Capture and Save the Image from the Canvas
        self.img = torch.tensor([np.array(ImageGrab.grab().crop((x, y, x1, y1)).convert("L"))],dtype=torch.float)
        print(self.img.shape)
        self.prediction.set("Prediction: " + str(int(self.model.predict(self.img)[0])))
        print(self.prediction)    

    def reset(self,e):    #reseting or cleaning the canvas 
        self.old_x = None
        self.old_y = None    
        self.save()  
        

    def clear(self):
        self.c.delete(ALL)
        self.prediction.set("Prediction")

    def drawWidgets(self):
    
        self.controls = Frame(self.master,padx = 5,pady = 5)
    
        l3 = Label(self.controls, textvariable = self.prediction,font=('arial 15'), anchor=NW)
        l3.grid(row=1,column=0)
        self.controls.pack(side=LEFT)
        
        self.c = Canvas(self.master,width=276,height=276,bg=self.color_bg)
        self.c.pack(fill=BOTH,expand=True)

        btn = Button ( self.master, command = self.clear,text="Clear")
        btn.place(x=0,y=50,width=100,height=30)

        menu = Menu(self.master)
        self.master.config(menu=menu)
 
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy) 
        
        

