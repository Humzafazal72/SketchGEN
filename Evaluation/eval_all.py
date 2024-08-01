# Code For Evaluating the entire test.  
# The Evaluation Metric Used is Multi Scale Structural Similarity Index Measure.

import os
import cv2
import numpy as np
from tqdm import tqdm
from sewar.full_ref import msssim

gen_path = 'D:\\SketchGen Code\\evaluation\\generated\\'
true_path = "D:\\SketchGen Data\\Final Dataset\\Test\\target\\"

img_names = sorted(os.listdir(true_path)) 

gen_img_paths = [gen_path+img_name for img_name in img_names]
true_img_paths = [true_path+img_name for img_name in img_names]

# checking all generated and true images have been aligned properly 

"""
for gen,true in zip(gen_img_paths,true_img_paths):
    if gen.split('\\')[4] != true.split('\\')[5]:
        print(gen)
        print(true)
        break
"""
all_msssims=[] 

for gen_path,true_path in tqdm(zip(gen_img_paths,true_img_paths)):
    gen = cv2.imread(gen_path,cv2.IMREAD_GRAYSCALE)
    true = cv2.imread(true_path,cv2.IMREAD_GRAYSCALE)
    true = cv2.resize(true,(256,256))

    score = np.real(msssim(gen,true))
    all_msssims.append(score)

avg_msssims = np.mean(all_msssims)
median_msssims = np.median(all_msssims)

print((f"Average Score: {str(avg_msssims)}"))
print(f"Median Score: {str(median_msssims)}")
      
with open('ms_ssim_scores.txt', 'w') as file:
    for score, name in tqdm(zip(all_msssims, img_names)):
        file.write(f"{name}: {str(score)}\n")

    file.write(f"Average Score: {str(avg_msssims)}\n")
    file.write(f"Median Score: {str(median_msssims)}\n")



    
