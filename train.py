import os
import torch
import torch.nn as nn
import imageio.v3 as iio
import numpy as np
#import matplotlib.pyplot as plt
from Seq2Seq import Seq2Seq
from torch.utils.data import DataLoader

data_dir = 'squashfs-root/dataset/'
os.system(data_dir)

n_channels = 3
frame_size = (160, 240)
n_frames = 22
n_train = 1000

train = np.zeros((n_train, n_frames, *frame_size, n_channels)) #(num_videos, num_frames,height, width, channels)

print('Loading train data...')
for i in range(n_train):
    for j in range(22):
        dir = f'../squashfs-root/dataset/train/video_{i}/image_{j}.png'
        img = iio.imread(dir)
        train[i][j] = img
print('train data loaded')

train = train.reshape(n_train, n_channels, n_frames, *frame_size)

def collate(batch):
    # scale pixels between 0 and 1
    batch = np.array(batch)
    batch = torch.FloatTensor(batch)
    batch = batch / 255.0                                           

    # first 11 frames as input, 22nd as target                    
    return batch[:,:,:11], batch[:,:,21].unsqueeze(1)    


train_loader = DataLoader(train, shuffle=True, 
                        batch_size=16, collate_fn=collate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(num_channels=3, num_kernels=64, 
kernel_size=(3, 3), padding=(1, 1), activation="relu", 
frame_size=(160, 240), num_layers=3).to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

num_epochs = 20

for epoch in range(1, num_epochs+1):
    train_loss = 0                                                 
    model.train()                                                  
    for batch_num, (input, target) in enumerate(train_loader, 1):  
        input, target = input.to(device), target.to(device)
        output = model(input)    

        optim.zero_grad()                                   
        loss = criterion(output.flatten(), target.flatten())       
        loss.backward()                                            
        optim.step()                                                                                        
        train_loss += loss.item()                                 
    train_loss /= len(train_loader.dataset)                       

    """
    val_loss = 0                                                 
    model.eval()                                                   
    with torch.no_grad():                                          
        for input, target in val_loader:                          
            output = model(input)                                   
            loss = criterion(output.flatten(), target.flatten())   
            val_loss += loss.item()                                
    val_loss /= len(val_loader.dataset)                            

    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))
    """
    print("Epoch:{} Training Loss:{:.2f}".format(epoch, train_loss))

