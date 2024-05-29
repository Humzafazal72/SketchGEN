import torch
from tqdm import tqdm
from train import DEVICE,L1_LAMBDA,L1_LOSS,BCE,gen,disc 
from DatasetLoader import test_loader

gen.eval()
disc.eval()

# Initialize running loss for generator and discriminator
running_loss_g = 0.0
running_loss_d = 0.0

with torch.no_grad():  # Disable gradient calculation
    for image, sketch in tqdm(test_loader):
        image, sketch = image.to(DEVICE), sketch.to(DEVICE)
        
        # Discriminator Evaluation
        D_real = disc(image, sketch)
        D_real_loss = BCE(D_real, torch.ones_like(D_real))
        
        fake = gen(image)
        D_fake = disc(image, fake)
        D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
        
        D_loss = (D_real_loss + D_fake_loss) / 2
        
        # Generator Evaluation
        D_fake = disc(image, fake)
        G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
        L1 = L1_LOSS(fake, sketch) * L1_LAMBDA
        G_loss = G_fake_loss + L1
        
        running_loss_d += D_loss.item()
        running_loss_g += G_loss.item()
        
# Calculate average losses
avg_loss_g = running_loss_g / len(test_loader)
avg_loss_d = running_loss_d / len(test_loader)

print('Generator Loss: ', avg_loss_g)
print('Discriminator Loss: ', avg_loss_d)