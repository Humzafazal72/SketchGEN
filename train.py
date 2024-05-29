import torch
import torch.nn as nn
from tqdm import tqdm
from DatasetLoader import train_loader
from Generator import Generator
from Discriminator import Discriminator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
L1_LAMBDA = 100
EPOCHS = 500

checkpoint = torch.load('/kaggle/input/epoch-401/p2p_401.pt')
disc = Discriminator(in_=6).to(DEVICE)
gen = Generator(in_=3, out_=64).to(DEVICE)
opt_disc = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999),)
opt_gen = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

disc.load_state_dict(checkpoint['disc_state_dict'])
gen.load_state_dict(checkpoint['gen_state_dict'])
opt_disc.load_state_dict(checkpoint['discopt_state_dict'])
opt_gen.load_state_dict(checkpoint['genopt_state_dict'])

BCE = nn.BCEWithLogitsLoss()
L1_LOSS = nn.L1Loss()

if __name__=='__main__':
    pass

for epoch in range(EPOCHS):
    running_loss_d = 0.0
    running_loss_g = 0.0
    for image,sketch in tqdm(train_loader):
        image,sketch = image.to(DEVICE),sketch.to(DEVICE)
        
        #Discriminator Training
        D_real = disc(image, sketch)
        D_real_loss = BCE(D_real, torch.ones_like(D_real))
        
        fake = gen(image)
        D_fake = disc(image, fake.detach())
        D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
        
        D_loss = (D_real_loss + D_fake_loss) / 2
        
        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()
        
        #Generator Training
        D_fake = disc(image, fake)
        G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
        L1 = L1_LOSS(fake, sketch) * L1_LAMBDA
        G_loss = G_fake_loss + L1
        
        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()  
        running_loss_d+=D_loss
        running_loss_g+=G_loss
        
    print('Generator Loss: ', running_loss_g/len(train_loader))
    print('Discriminator Loss: ', running_loss_d/len(train_loader))