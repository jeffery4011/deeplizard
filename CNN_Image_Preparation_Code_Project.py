import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt

# class OHLC(Dataset):
#     def __init__(self,csv_file):
#         self.data= pd.read_csv(csv_file)

#     def __getitem__(self,index):
#         r = self.data.iloc[index]
#         label = torch.tensor(r.is_up_day,dtype=torch.long)
#         sample = self.normalize(torch.tenser([r.open,r.high,r.low,r.close]))
#         return sample, label

#     def __len__(self):
#         return len(self.data)


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train =True
    ,download=True
    ,transform=transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
)

train_loader = torch.utils.data.DataLoader(train_set
    ,batch_size=100
    # ,shuffle=True
)

torch.set_printoptions(linewidth=120)

sample = next(iter(train_set))
image, label = sample


batch = next(iter(train_loader))
images, labels = batch

grid = torchvision.utils.make_grid(images, nrow=10)
print('labels:',labels)
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid,(1,2,0)))
plt.pause(10)
