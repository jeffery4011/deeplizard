import torch
import torch.nn as nn

class Network:
    def __int__(self):
        super(Network.self)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)


    def forward(self,t):
        return t

fc = nn.Linear(in_features=4,out_features=3,bias=False)
weight_matrix = torch.tensor([
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6]
], dtype=torch.float32)
t= torch.tensor([1,2,3,4],dtype=torch.float32)
fc.weight=nn.Parameter(weight_matrix)
print(fc(t))