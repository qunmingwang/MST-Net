# Propose, 时间序列影像厚云
# 输入数据格式tif,每次重建cloudy 1波段,使用4个temporal的对应波段
import torch 
import torch.nn as nn
import numpy as np
from osgeo import gdal
import time
import itertools
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch import linalg as LA
import tifffile as tiff

Tnum = 7 
lr = 0.001 # defualt=0.001
num_epochs = 60 # defualt=100
batch_size = 16 

bands = 4 
lam = 0.15
rd = 20 
sub_size = 50 

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cloudys1 = gdal.Open('.\MST/MS{}pre1_100e.tif'.format(Tnum)) 
cloudys2 = gdal.Open('.\MST/MS{}pre2_100e.tif'.format(Tnum))
cloudys1 = cloudys1.ReadAsArray()
cloudys1 = torch.from_numpy(cloudys1.astype(np.float32)).clone()
cloudys2 = cloudys2.ReadAsArray()
cloudys2 = torch.from_numpy(cloudys2.astype(np.float32)).clone()
cloudys = torch.cat((cloudys1,cloudys2),dim=0) 

knownS = gdal.Open('./train5000_closeS\Global_new2\S10m6bz{}.tif'.format(Tnum)) # S2 10m bands
knownS = knownS.ReadAsArray()
knownS = torch.from_numpy(knownS.astype(np.float32)).clone()
known2 = gdal.Open('./train5000_closeS\Global_new2/L1_{}.tif'.format(Tnum))
known2 = known2.ReadAsArray()
known2 = torch.from_numpy(known2.astype(np.float32)).clone()
known3 = gdal.Open('./train5000_closeS\Global_new2/L2_{}.tif'.format(Tnum))
known3 = known3.ReadAsArray()
known3 = torch.from_numpy(known3.astype(np.float32)).clone()
known4 = gdal.Open('./train5000_closeS\Global_new2/L3_{}.tif'.format(Tnum))
known4 = known4.ReadAsArray()
known4 = torch.from_numpy(known4.astype(np.float32)).clone()

mask1 = gdal.Open('./train5000_closeS\Global_new2/mask{}_1_1500.tif'.format(Tm))
mask1 = mask1.ReadAsArray()
mask1 = torch.from_numpy(mask1.astype(np.float32)).clone()
mask2 = gdal.Open('./train5000_closeS\Global_new2/mask{}_2.tif'.format(Tm))
mask2 = mask2.ReadAsArray()
mask2 = torch.from_numpy(mask2.astype(np.float32)).clone()
mask3 = gdal.Open('./train5000_closeS\Global_new2/mask{}_3.tif'.format(Tm))
mask3 = mask3.ReadAsArray()
mask3 = torch.from_numpy(mask3.astype(np.float32)).clone()
mask4 = gdal.Open('./train5000_closeS\Global_new2/mask{}_4.tif'.format(Tm))
mask4 = mask4.ReadAsArray()
mask4 = torch.from_numpy(mask4.astype(np.float32)).clone()

mask_knownLs = torch.stack((mask2,mask3,mask4)) # layer stacking
mask_Knon = torch.ones(mask_knownLs.size())-mask_knownLs
mask_KSnon = torch.ones(mask1.size())-mask1

# knownS = (knownS-1000)/10000
known2 = known2*0.0000275-0.2
known3 = known3*0.0000275-0.2
known4 = known4*0.0000275-0.2

known1 = knownS*mask_KSnon
known2 = known2*mask_Knon[0]
known3 = known3*mask_Knon[1]
known4 = known4*mask_Knon[2]

###### Define neural network
# 3x3 convolution
def conv3x3Down(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=3,
                     padding=0, bias=False)

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=1, bias=False)

def conv5x5(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, 
                     padding=2, bias=False)

def conv7x7(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, 
                     padding=3, bias=False)

# Network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv_downscale = conv3x3Down(1,1)
        self.conv_c1 = conv3x3(1,20)
        self.conv_c2 = conv5x5(1,20)
        self.conv_c3 = conv7x7(1,20)
        self.conv_k1 = conv3x3(4,20)
        self.conv_k2 = conv5x5(4,20)
        self.conv_k3 = conv7x7(4,20)
        self.conv_r1 = nn.Sequential(conv3x3(120,60),nn.ReLU(True))
        self.conv_r2 = nn.Sequential(conv3x3(60,60),nn.ReLU(True))
        self.conv_r3 = nn.Sequential(conv3x3(60,60),nn.ReLU(True))
        self.conv_r4 = nn.Sequential(conv3x3(60,60),nn.ReLU(True))
        self.conv_r5 = nn.Sequential(conv3x3(60,60),nn.ReLU(True))
        self.conv_r6 = nn.Sequential(conv3x3(60,60),nn.ReLU(True))
        self.conv_r7 = nn.Sequential(conv3x3(60,60),nn.ReLU(True))
        self.conv_r8 = nn.Sequential(conv3x3(60,60),nn.ReLU(True))
        self.conv_r9 = nn.Sequential(conv3x3(60,60),nn.ReLU(True))
        self.conv_r10 = nn.Sequential(conv3x3(60,30),nn.ReLU(True))

        self.conv = conv3x3(30,1)

    def forward(self, MSpre, k, ks):
        x1 = self.conv_c1(MSpre)
        x2 = self.conv_c2(MSpre)
        x3 = self.conv_c3(MSpre)
        ys = self.conv_downscale(ks)
        yall = torch.cat((ys,k),dim=1)
        y1 = self.conv_k1(yall)
        y2 = self.conv_k2(yall)
        y3 = self.conv_k3(yall)
        y = torch.cat((x1,x2,x3,y1,y2,y3),dim=1)

        out = self.conv_r1(y)
        out = self.conv_r2(out)
        out = self.conv_r3(out)
        out = self.conv_r4(out)
        out = self.conv_r5(out)
        out = self.conv_r6(out)
        out = self.conv_r7(out)
        out = self.conv_r8(out)
        out = self.conv_r9(out)
        out = self.conv_r10(out)

        out = self.conv(out)
        return out

model = Net()
# model.load_state_dict(torch.load('./train5000_closeS\Global_results/MTnet_4k_b2_0.001_20e_16b.pth'))

for bn in range(2,8): 
    cloudy02D = cloudys[bn-2] 
    cloudy = torch.unsqueeze(cloudy02D,0) 
    cloudy = torch.unsqueeze(cloudy,0) 

    k2D =knownS[bn-2] 
    k2D = torch.unsqueeze(k2D,0)
    kS = torch.unsqueeze(k2D,0)

    known = torch.stack((known2[bn-2],known3[bn-2],known4[bn-2])) # layer stacking
    known = torch.unsqueeze(known,0) 
    
    for epoch in range(num_epochs):
        if (epoch+1) % 60 == 0: 
            model.load_state_dict(torch.load('./train5000_closeS\MSTnet_Global/MTnet_4k_b{}_{}_{}e_{}b.pth'.format(bn, lr, epoch+1, batch_size)))
            output = model(cloudy,known,kS) 
            pre = torch.squeeze(output,0) 
            pre = torch.squeeze(pre,0)
            pre = pre.detach().numpy() 
            tiff.imsave('./train5000_closeS\Global_new2\MST/MSTpre{}_b{}_{}_{}_{}.tif'.format(Tnum,bn, lr, epoch+1, batch_size),pre) # 输出tif格式预测

# python MT-Net.py