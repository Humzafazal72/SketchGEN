import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader,Dataset

class ImageDataset(Dataset):
    def __init__(self, src_dir, tgt_dir):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.list_files = os.listdir(self.src_dir)
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        
        src_path = os.path.join(self.src_dir, img_file)
        tgt_path = os.path.join(self.tgt_dir, img_file)
        
        # Load images and convert to RGB
        src_image = Image.open(src_path).resize((256, 256)).convert('RGB')
        tgt_image = Image.open(tgt_path).resize((256, 256)).convert('RGB')
        
        # Convert images to numpy arrays and scale pixel values to [0, 1]
        src_image = np.array(src_image) / 255.0
        tgt_image = np.array(tgt_image) / 255.0
        
        # Transpose the arrays to change shape from (256, 256, 3) to (3, 256, 256)
        src_image = np.transpose(src_image, (2, 0, 1))
        tgt_image = np.transpose(tgt_image, (2, 0, 1))
        
        # Convert numpy arrays to torch tensors
        src_image = torch.tensor(src_image, dtype=torch.float32)
        tgt_image = torch.tensor(tgt_image, dtype=torch.float32)
        
        return src_image, tgt_image
    
train_dataset = ImageDataset(src_dir='',tgt_dir='')
test_dataset = ImageDataset(src_dir='',tgt_dir='')

train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=4) 
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)