# How to take inference and save sketch
import torch
import numpy as np
from train import gen,DEVICE
import matplotlib.pyplot as plt
from DatasetLoader import test_loader

c = 0 #number of pictures
g = []
gen.eval()
with torch.no_grad():
    for image,sk in test_loader:
        c+=1
        if c<50:
            g.append(gen(image.to(DEVICE)))
            continue
        break

preds = []
for img in g:
    pred = np.array(img.detach().to('cpu'))
    pred = pred[0].transpose(1, 2, 0)
    preds.append(pred)

i=0
for pred in preds:
    pred = np.clip(pred, 0.0, 1.0)
    plt.imsave(f'generated{i}.jpg', pred)
    i+=1