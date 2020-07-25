import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms




torch.set_printoptions(linewidth=120)

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


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

@torch.no_grad()
def get_all_preds(model,loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images , labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds,preds),
            dim=0
        )
    return all_preds



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

network = Network()
prediction_loader = torch.utils.data.DataLoader(train_set, batch_size = 10000)
train_preds = get_all_preds(network, prediction_loader)

preds_correct = get_num_correct(train_preds,train_set.targets)
print('accuracy:',preds_correct/len(train_set))