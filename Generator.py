import torch
import troch.nn as nn

def generator_block(in_,out_,down=True,act='relu',dropout=False):
    layers = []
    
    if not down:
        layers.append(nn.ConvTranspose2d(in_,out_,kernel_size=4,stride = 2, padding=1, bias=False))
    else:
        layers.append(nn.Conv2d(in_,out_,kernel_size=4,stride = 2, padding=1, bias=False,padding_mode='reflect'))
    
    layers.append(nn.BatchNorm2d(out_))
    
    if act == 'relu':
        layers.append(nn.ReLU())
    else:
        layers.append(nn.LeakyReLU(0.2))
    
    if not dropout:
        return nn.Sequential(*layers)
    layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers) 
    
class Generator(nn.Module):
    def __init__(self,in_=3,out_=64):
        super().__init__()
        # Encoder
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_, out_, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.d1 = generator_block(out_,out_*2,act='leaky')
        self.d2 = generator_block(out_*2,out_*2*2,act='leaky')
        self.d3 = generator_block(out_*2*2,out_*2*2*2,act='leaky')
        self.d4 = generator_block(out_*2*2*2,out_*2*2*2,act='leaky')
        self.d5 = generator_block(out_*2*2*2,out_*2*2*2,act='leaky')
        self.d6 = generator_block(out_*2*2*2,out_*2*2*2,act='leaky')
        
        # BottleNeck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_*2*2*2, out_*2*2*2, 4, 2, 1), 
            nn.ReLU()
        )
        
        #Decoder 
        self.u1 = generator_block(out_ * 8, out_ * 8, down=False, act="relu", dropout=True)
        self.u2 = generator_block(out_ * 8 * 2, out_ * 8, down=False, act="relu", dropout=True)
        self.u3 = generator_block(out_ * 8 * 2, out_ * 8, down=False, act="relu", dropout=True)
        self.u4 = generator_block(out_ * 8 * 2, out_ * 8, down=False, act="relu", dropout=False)
        self.u5 = generator_block(out_ * 8 * 2, out_ * 4, down=False, act="relu", dropout=False)
        self.u6 = generator_block(out_ * 4 * 2, out_ * 2, down=False, act="relu", dropout=False)
        self.u7 = generator_block(out_ * 2 * 2, out_, down=False, act="relu", dropout=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(out_ * 2, in_, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    def forward(self,x):
        d1 = self.initial_down(x)
        d2 = self.d1(d1)
        d3 = self.d2(d2)
        d4 = self.d3(d3)
        d5 = self.d4(d4)
        d6 = self.d5(d5)
        d7 = self.d6(d6)
        
        bottleneck = self.bottleneck(d7)
        
        u1 = self.u1(bottleneck)
        u2 = self.u2(torch.cat([u1, d7], 1))
        u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1))
        u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1))
        u7 = self.u7(torch.cat([u6, d2], 1))
        
        return self.final_up(torch.cat([u7, d1], 1))