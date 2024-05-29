import torch
import torch.nn as nn

def discriminator_block(in_,out_,stride):
    return nn.Sequential(
        nn.Conv2d(in_,out_,kernel_size=4,stride=stride,padding=1,bias=False,padding_mode='reflect'),
        nn.BatchNorm2d(out_),
        nn.LeakyReLU(0.2)
    )

class Discriminator(nn.Module):
    def __init__(self,in_=6,out_=64):
        super().__init__()
        self.initial_block = nn.Sequential(
                nn.Conv2d(in_,out_,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),#** in_ size in video =6
                nn.LeakyReLU(0.2)
        )
        self.main_body = nn.Sequential(
                discriminator_block(out_,out_*2,stride =2),
                discriminator_block(out_*2,out_*2*2,stride =2),
                discriminator_block(out_*2*2,out_*2*2*2,stride =1),
                nn.Conv2d(out_*2*2*2, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        ) 
    
    def forward(self,x,y):
        x = torch.cat([x, y], dim=1)
        x = self.initial_block(x)
        return self.main_body(x)
    
if __name__=='__main__':
    pass
        