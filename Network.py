import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)

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
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)



    

    def forward(self,t):
        #(1)input layer
        t=t
        
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)   
        t = F.max_pool2d(t, kernel_size=2, stride=2)# find the max element in 2*2 matrix with stepsize 2 stride. 

        #(3)hidden conv Layer
        t = self.conv2(t)
        t = F.relu(t)# relu function :max(t,0)
        t = F.max_pool2d(t,kernel_size=2,stride=2)

        #(4) hidden Linear layer
        t = t.reshape(-1,12*4*4)
        t= self.fc1(t)
        t= F.relu(t)

        #(5) hidden Linear Layer
        t = self.fc2(t)
        t = F.relu(t)

        #(6) output layer
        t = self.out(t)
        #t = F.softmax(t,dim=1)

        return t

torch.set_grad_enabled(False)

network = Network()

sample = next(iter(train_set))

image,label =sample
print (image.shape)

pred = network(image.unsqueeze(0))#image shape nees to be (batch_size in_channels. height, width)
print(pred.shape)

print(F.softmax(pred,dim=1))