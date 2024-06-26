# How to take inference and save sketch
import cv2
import torch
import numpy as np
from train import gen,DEVICE
import matplotlib.pyplot as plt

def get_inference(src_path):
    img = cv2.imread(src_path)
    img = cv2.resize(img,(256,256))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = np.transpose(img, (2, 0, 1))/255.0
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img,dtype=torch.float32)

    # use your trained Generator or our trained weights.
    gen.eval()
    with torch.no_grad():
        pred = gen(img)
    
    pred = np.array(img.detach().to('cpu'))
    pred = pred[0].transpose(1, 2, 0)

    pred = np.clip(pred, 0.0, 1.0) #clip the negative pixels 
    plt.imsave(r'generated.jpg', pred)

if __name__ == '__main__':
    get_inference(src_path)
