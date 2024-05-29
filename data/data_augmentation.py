import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def apply_augmentation(src_path, dst_path):
    source_folder = src_path
    files = sorted(os.listdir(source_folder))
    for filename in tqdm(files):
        path = os.path.join(source_folder, filename)
        img = plt.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # mirror image
        img_flip0 = cv2.flip(img, flipCode=0)

        # flip image upside down
        img_flip1 = cv2.flip(img, flipCode=1)

        # rotate by 270
        rotated_img_270 = cv2.rotate(img, rotateCode=0)

        # rotate by 45
        rotation_angle = 45
        rows, cols, _ = img.shape
        rotation_matrix = cv2.getRotationMatrix2D(
            (cols / 2, rows / 2), rotation_angle, 1)
        rotated_img_45 = cv2.warpAffine(img, rotation_matrix, (cols, rows))

        # rotate by 90
        rotated_img_90 = cv2.rotate(img, rotateCode=2)
        dst = dst_path
        name = str(dst + str(filename.replace('.jpg', '')))

        cv2.imwrite(img=img_flip0, filename=name+'-0.jpg')
        cv2.imwrite(img=img_flip1, filename=name+'-1.jpg')
        cv2.imwrite(img=rotated_img_45, filename=name+'-2.jpg')
        cv2.imwrite(img=rotated_img_90, filename=name+'-3.jpg')
        cv2.imwrite(img=rotated_img_270, filename=name+'-4.jpg')


if __name__ == '__main__':
    #apply augmentation to Portraits
    apply_augmentation(src_path=r"..\\mock data\\portraits\\",
                       dst_path=r"destination_path_here\\")
    #apply augmentation to Sketches
    apply_augmentation(src_path=r"..\\mock data\\sketches\\",
                       dst_path=r"destination_path_here\\")
