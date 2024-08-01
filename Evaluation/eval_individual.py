import cv2
import numpy as np
import matplotlib.pyplot as plt
from sewar.full_ref import msssim

                    #Load images (they should be in grayscale for SSIM)
imageA = cv2.imread('D:\\SketchGen Code\\evaluation\\generated\\100.jpg',cv2.IMREAD_GRAYSCALE)
imageB = cv2.imread("D:\\SketchGen Data\\Final Dataset\\Test\\target\\100.jpg",cv2.IMREAD_GRAYSCALE)
imageB = cv2.resize(imageB,(256,256))

                    #Display Images
fig,ax = plt.subplots(1,2)
ax[0].imshow(imageA,cmap='gray')
ax[0].set_title('Generated Image')

ax[1].imshow(imageB,cmap='gray')
ax[1].set_title('True/Original Image')

plt.show()

            # Compute MS_SSIM between two images
print(f"MS_SSIM: {np.real(msssim(imageA,imageB))}")