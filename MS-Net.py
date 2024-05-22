# MS-Net
# 输入数据格式为tif,按波段排成行向量的无云区数据
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
import numpy as np
from osgeo import gdal
import time
import matplotlib.pyplot as plt
import tifffile as tiff
from sklearn.model_selection import train_test_split

# Hyper parameters
Tnum = 10 
batch_size = 256
num_epochs = 100 # defualt=100
learning_rate = 0.1 # defualt=0.1
rd = 60

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dataset
S2  = gdal.Open('./train5000_closeS\MSTnet_Global/test500/S10mz{}.tif'.format(Tnum)) 
L8 = gdal.Open('./train5000_closeS\Global_newarea/Lc{}.tif'.format(Tnum))
mask = gdal.Open('./train5000_closeS\Global_newarea/new4Region/maskC{}.tif'.format(Tnum))
mask1 = gdal.Open('./train5000_closeS\Global_newarea/new4Region/Region{}_mask1.tif'.format(Tnum))

S2 = S2.ReadAsArray()
S2 = torch.from_numpy(S2.astype(np.float32)).clone()
L8 = L8.ReadAsArray()
L8 = torch.from_numpy(L8.astype(np.float32)).clone()
mask = mask.ReadAsArray()
mask = torch.from_numpy(mask.astype(np.float32)).clone()
mask1 = mask1.ReadAsArray()
mask1 = torch.from_numpy(mask1.astype(np.float32)).clone()

# S2 = (S2-1000)/10000
L8 = L8*0.0000275-0.2
# mask = mask/255 

# test subarea
line = 500 
bands = 6
subS = 3 
W = line*3 
H = line*3 

w =  int(W/subS) 
h =  int(H/subS)

maskall = mask + mask1 
one = torch.ones(maskall.size())
mask_non = one - maskall
mask_non = mask_non.repeat(bands,1,1) 

oneC = torch.ones(mask.size())
maskC_non = oneC - mask # 1-非云区的mask
maskC_non = maskC_non.repeat(bands,1,1) 

# divide 
S2_Line = torch.chunk(S2, w, dim=2) 
S2_L = torch.cat([fm.unsqueeze(0) for fm in S2_Line], dim=0)
S2_Lh = torch.chunk(S2_L, h, dim=2) 
S2_LH = torch.cat([fm.unsqueeze(0) for fm in S2_Lh], dim=0) 
S2_chop = S2_LH.reshape(w*h,bands,subS,subS) 

L8_Line = torch.chunk(L8, w, dim=2)
L8_L = torch.cat([fm.unsqueeze(0) for fm in L8_Line], dim=0)
L8_Lh = torch.chunk(L8_L, h, dim=2)
L8_LH = torch.cat([fm.unsqueeze(0) for fm in L8_Lh], dim=0)
L8_chop = L8_LH.reshape(w*h,bands)

masknon_Line = torch.chunk(mask_non, w, dim=2)
masknon_L = torch.cat([fm.unsqueeze(0) for fm in masknon_Line], dim=0)
masknon_Lh = torch.chunk(masknon_L, h, dim=2)
masknon_LH = torch.cat([fm.unsqueeze(0) for fm in masknon_Lh], dim=0)
masknon_chop = masknon_LH.reshape(w*h,bands)

Train = TensorDataset(S2_chop, L8_chop, masknon_chop) 

train_set, test_set = train_test_split(Train, test_size=0.1, random_state=42) # random_state:设置随机种子，保证每次运行生成相同的随机数
train_iter = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False) 

###### Define neural network
# 3x3 convolution for 10m to 30m
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=3,
                     padding=0, bias=False)

# AutoEncoder net
class Net(nn.Module):
    def __init__(self, bands=bands):
        super(Net, self).__init__()
        self.conv = conv3x3(bands,bands) 
        self.fc1 = nn.Linear(bands, bands*5) # hidden = 5*bands
        self.rl1 = nn.ReLU(True)
        self.fc2 = nn.Linear(bands*5, bands*10)
        self.rl2 = nn.ReLU(True)
        self.fc3 = nn.Linear(bands*10, bands*5)
        self.rl3 = nn.ReLU(True)
        self.fc4 = nn.Linear(bands*5, bands)
        
    def forward(self, x):
        out = self.conv(x)
        out = torch.squeeze(out,3) 
        out = torch.squeeze(out,2) 
        out = self.fc1(out)
        out = self.rl1(out)
        out = self.fc2(out)
        out = self.rl2(out)
        out = self.fc3(out)
        out = self.rl3(out)
        out = self.fc4(out)
        return out

model = Net(bands).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

###### Train the model ######
total_step = len(train_iter)
total_test_step = len(test_iter)
print ('mission-->batch size:{}, epoch:{}, step:{}'.format( batch_size, num_epochs, total_step))
curr_lr = learning_rate
loss_list = []
loss_list_test = []
start_time=time.time()

for epoch in range(num_epochs):
    losslist = 0
    losslist_test = 0
    for i, (S2s, L8s, masknons) in enumerate(train_iter):
        S2s = S2s.to(device)
        L8s = L8s.to(device)
        masknons = masknons.to(device)

        # Forward pass
        outputs = model(S2s)
        S2s_refer = outputs*masknons
        L8s_refer = L8s*masknons
        loss = criterion(S2s_refer, L8s_refer)
        losslist += loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for ii, (S2s_test, L8s_test, masknons_test) in enumerate(test_iter):
        S2s_test = S2s_test.to(device)
        L8s_test = L8s_test.to(device)
        masknons_test = masknons_test.to(device)

        # testing loss
        outputs_test = model(S2s_test)
        S2s_test_refer = outputs_test*masknons_test
        L8s_test_refer = L8s_test*masknons_test
        loss_test = criterion(S2s_test_refer, L8s_test_refer)
        losslist_test += loss_test

    if (epoch+1) % 1 == 0:
        print ('Epoch [{}/{}],  Loss: {:.8f},  Loss_test: {:.8f}' 
                .format(epoch+1, num_epochs, loss.item(), loss_test.item()))
    
    loss_list.append(losslist/total_step)
    loss_list_test.append(losslist_test/total_test_step )

    if (epoch+1) % 100 == 0:
        torch.save(model.state_dict(), '.\MST/MSnet{}_{}lr_{}e_{}b.pth'.format(Tnum, learning_rate, epoch+1, batch_size))
 
    # Decay learning rate
    if (epoch+1) % rd == 0:
        curr_lr *= 0.95
        update_lr(optimizer, curr_lr)

print('total time: {:.4f} min'.format((time.time() - start_time)/60))

###### predict ######
print('test data preparing')
S2_chop = S2_chop.to(device) 
mask = mask.detach().numpy()
maskC_non = maskC_non.detach().numpy()
for epoch in range(num_epochs):
    if (epoch+1) % 100 == 0:
        model.load_state_dict(torch.load('.\MST/MSnet{}_{}lr_{}e_{}b.pth'.format(Tnum, learning_rate, epoch+1, batch_size)))
        pre = model(S2_chop) # S2 to L8 全图
        pre = pre.reshape(line,-1,bands)
        pre = pre.permute(2,0,1) # 维度顺序转换 ##(channel, x, y)
        pre = pre.cpu()  # 将tensor复制到主机中   
        pre = pre.detach().numpy()  # tensor转numpy格式,从而存储
        c = L8.detach().numpy()
        Lfilled = c*maskC_non + pre*mask 
        LF1 = Lfilled[:3]
        LF2 = Lfilled[3:]
        tiff.imsave('.\MST/MS{}pre1_{}e.tif'.format(Tnum,epoch+1),LF1) # 输出band1,2,3
        tiff.imsave('.\MST/MS{}pre2_{}e.tif'.format(Tnum,epoch+1),LF2)

# pain the loss line chart of training
x = range(num_epochs)
y = torch.tensor(loss_list, device = 'cpu') # list类型放cpu
plt.subplot(2, 1, 1)
plt.plot(x, y, 'o-')
plt.title('lr:{}_rd:{}'.format(learning_rate, rd))
plt.xlabel('epoch')
plt.ylabel('loss')

# loss line chart of training
x_test = range(num_epochs)
y_test = torch.tensor(loss_list_test, device = 'cpu') # list类型放cpu
plt.subplot(2, 1, 2)
plt.plot(x_test, y_test, 'o-')
plt.title('lr:{}_rd:{}'.format(learning_rate, rd))
plt.xlabel('epoch')
plt.ylabel('loss_test')
plt.show()

# python MS-Net.py