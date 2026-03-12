import torch
import argparse
import torch.nn as nn
import torch.utils
from tqdm import tqdm
from pathlib import Path
from DatasetLoader import create_dataloader
from architecture import unetGenerator,Discriminator,unetppGenerator

# disc.load_state_dict(checkpoint['disc_state_dict'])
# gen.load_state_dict(checkpoint['gen_state_dict'])
# opt_disc.load_state_dict(checkpoint['discopt_state_dict'])
# opt_gen.load_state_dict(checkpoint['genopt_state_dict'])

def training_loop(EPOCHS:int, DEVICE:torch.device, L1_LAMBDA:int, train_loader:torch.utils.data.DataLoader):
    for epoch in range(EPOCHS):
        running_loss_d = 0.0
        running_loss_g = 0.0
        for image,sketch in tqdm(train_loader,desc=f"EPOCH {epoch+1}:"):
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
            running_loss_d+=D_loss.item()
            running_loss_g+=G_loss.item()

        print('Generator Loss: ', running_loss_g/len(train_loader))
        print('Discriminator Loss: ', running_loss_d/len(train_loader))
    
    Path('model_weights').mkdir(parents=True, exist_ok=True)

    torch.save({
        'gen_state_dict': gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'genopt_state_dict': opt_gen.state_dict(),
        'discopt_state_dict': opt_disc.state_dict(),
        'epoch': epoch
    }, f"model_weights/model_epoch_{epoch+1}.pth")
    print(f"Weights saved at model_weights/model_epoch_{epoch+1}.pth")
    

if __name__=='__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dir", help="provide the path to portraits",
                        default='data\mock dataset\portrait', type=str)
    parser.add_argument("-t", "--tgt_dir", help="provide the path to sketches",
                        default='data\mock dataset\sketches', type=str)
    parser.add_argument("-g", "--gen_type", help="provide the name of the generator",
                        default='unet', type=str, choices=['unet','unet++'])
    parser.add_argument("-e", "--epochs", help="provide the number of epochs",
                        default=500, type=int)

    args = parser.parse_args()
    EPOCHS = args.epochs
    src_dir = Path(args.src_dir)
    tgt_dir = Path(args.tgt_dir)
    gen_type = args.gen_type

    if src_dir.exists() == False:
        ValueError(f"{src_dir} does not exist. Please provide valid path.")
        exit()
    
    if tgt_dir.exists() == False:
        ValueError(f"{tgt_dir} does not exist. Please provide valid path.")
        exit()

    
    train_loader = create_dataloader(src_dir=src_dir,tgt_dir=tgt_dir)

    BCE = nn.BCEWithLogitsLoss()    
    L1_LOSS = nn.L1Loss()

    disc = Discriminator(in_=6).to(DEVICE)
    if gen_type=='unet':
        gen = unetGenerator(in_=3, out_=64).to(DEVICE)
    else:
        gen = unetppGenerator.to(DEVICE)

    print(gen)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_gen = torch.optim.Adam(gen.parameters(), lr=100, betas=(0.5, 0.999))

    training_loop(EPOCHS = EPOCHS, DEVICE=DEVICE, L1_LAMBDA = 100, train_loader=train_loader)
