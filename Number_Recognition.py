
from CNN import * 
from Drawing_Board import *
from IntGrad import *

model_path = "./model/mnist-cnn.pth"

cnn = CNN()
cnn.load_state_dict(torch.load(model_path))
cnn.eval()

ig = IntegratedGradient(cnn) 

if __name__ == '__main__':
    root = Tk()
    GUI(root,cnn, ig)
    root.title('Number Recognition')
    root.mainloop()

