import torch
import torch.nn as nn
from Discriminator import Discriminator
from Generator import Generator

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