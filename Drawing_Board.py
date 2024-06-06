from tkinter import *
from PIL import ImageGrab, Image, ImageTk 
import numpy as np
import torch
import torchvision.transforms as T
from IntGrad import *
import matplotlib.pyplot as plt

class GUI:
    def __init__(self, master, model, explainer):
        self.prediction = StringVar()
        self.prediction.set("Prediction")
        self.model = model
        self.explainer = explainer
        self.img = None
        self.saliency_map = ImageTk.PhotoImage(T.ToPILImage()(torch.ones((284, 284))))
        self.master = master
        self.color_fg = '#ffffff'
        self.color_bg = '#000000'
        self.old_x = None
        self.old_y = None
        self.penwidth = 30
        
        self.drawWidgets()
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def paint(self, e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, e.x, e.y, width=self.penwidth, fill=self.color_fg, capstyle=ROUND, smooth=False)

        self.old_x = e.x
        self.old_y = e.y

    def save(self):
        x = self.c.winfo_rootx()
        y = self.c.winfo_rooty()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        self.img = torch.tensor([np.array(ImageGrab.grab().crop((x, y, x1, y1)).convert("L"))], dtype=torch.float) / 255.0
        self.prediction.set("Prediction: " + str(int(self.model.predict(self.img)[0])))
        
        self.draw_bar_chart(self.model.probability_array(self.img))
        #print(self.model.probability_array(self.img))
        bar_chart_image = Image.open('bar_chart.png')
        bar_chart_image = ImageTk.PhotoImage(bar_chart_image)
        self.bar_chart_label.configure(image=bar_chart_image)
        self.bar_chart_label.image = bar_chart_image  # Keep a reference to avoid garbage collection
   

    def reset(self, e):
        self.old_x = None
        self.old_y = None
        self.save()

    def explain(self):
        img = torchvision.transforms.Resize((28,28))(self.img)
        self.saliency_map = self.explainer.explain(img, step=100).squeeze()
        self.saliency_map = T.ToPILImage()(self.saliency_map)
        self.saliency_map = T.Resize((284, 284))(self.saliency_map)
        self.saliency_map = ImageTk.PhotoImage(self.saliency_map)
        self.explanation_label.configure(image=self.saliency_map)
        self.explanation_label.image = self.saliency_map

    def clear(self):
        self.c.delete(ALL)
        self.prediction.set("Prediction")
        
        self.saliency_map = ImageTk.PhotoImage(T.ToPILImage()(torch.ones((284, 284))))
        self.explanation_label.configure(image=self.saliency_map)
        self.explanation_label.image = self.saliency_map
        
        probabilities = [0] * 10
        self.draw_bar_chart(probabilities)
        bar_chart_image = Image.open('bar_chart.png')
        bar_chart_image = ImageTk.PhotoImage(bar_chart_image)
        self.bar_chart_label.configure(image=bar_chart_image)
        self.bar_chart_label.image = bar_chart_image  

    def draw_bar_chart(self, probabilities):
        digits = [str(i) for i in range(10)]
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.ylim(0,1)
        plt.title('Digit Probabilities')
        plt.bar(digits, probabilities, color='blue')
        plt.savefig('bar_chart.png')  # Save the bar chart as an image
        plt.close()

    def drawWidgets(self):
        # Main Frame
        self.main_frame = Frame(self.master)
        self.main_frame.pack(fill=BOTH, expand=False)

        #Top Stack Frame
        self.top_frame = Frame(self.main_frame, padx=10, pady=10)
        self.top_frame.pack(side = TOP, fill=BOTH, expand=False)

        # Canvas
        self.c = Canvas(self.top_frame, width=280, height=280, bg=self.color_bg)
        self.c.pack(side=LEFT, fill='none', expand=False)

        # Controls Frame
        self.controls_frame = Frame(self.top_frame, padx=10, pady=10)
        self.controls_frame.pack(fill=BOTH, expand=False)

        # Predict Frame
        self.predict_frame = Frame(self.controls_frame, padx=10, pady=10)
        self.predict_frame.pack(side=LEFT, fill=BOTH, expand=False)

        # Button Frame
        self.button_frame = Frame(self.predict_frame, padx=10, pady=10)
        self.button_frame.pack(side=BOTTOM, fill=BOTH, expand=False)

        # Explanation Image Placeholder
        explanation_image = self.saliency_map  # Path to your image
        self.explanation_label = Label(self.controls_frame, image=explanation_image)
        self.explanation_label.image = explanation_image  # Keep a reference to avoid garbage collection
        self.explanation_label.pack(pady=10, side=RIGHT)

        # Prediction Label
        l3 = Label(self.predict_frame, textvariable=self.prediction, font=('Arial', 15), width=10, justify='left')
        l3.pack(side=TOP, pady=10)

        # Clear Button
        btn = Button(self.button_frame, text="Clear", command=self.clear)
        btn.pack(side=LEFT, pady=10, padx=10)

        # Explainer Button
        btn2 = Button(self.button_frame, text="Explain", command=self.explain)
        btn2.pack(side=RIGHT, pady=10, padx=10)

        # Example bar chart data
        probabilities = [0] * 10
        self.draw_bar_chart(probabilities)

        # Bar Chart Placeholder Image
        bar_chart_image = Image.open('bar_chart.png')
        bar_chart_image = ImageTk.PhotoImage(bar_chart_image)
        self.bar_chart_label = Label(self.main_frame, image=bar_chart_image)
        self.bar_chart_label.image = bar_chart_image  # Keep a reference to avoid garbage collection
        self.bar_chart_label.pack(pady=10, side=BOTTOM)



