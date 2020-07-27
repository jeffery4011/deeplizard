import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from plot_cm import plot_confusion_matrix

from collections import OrderedDict
from collections import namedtuple
import itertools
from itertools import product
import numpy as np

from torch.utils.tensorboard import SummaryWriter

batch_size_list =[100,1000,10000]
lr_list = [0.01, 0.0001, 0.00001]

parameters = dict(
    lr =[0.01, 0.0001, 0.00001],
    batch_size = [10,100,1000,10000],
    shuffle = [True,False]
)
param_values = [v for v in parameters.values()]

params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [1000,10000]
    ,shuffle = [True,False]
)

Run = namedtuple('Run',params.keys())
torch.set_printoptions(linewidth=120)
runs = []
for v in product(*params.values()):
    runs.append(Run(*v))

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

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run',params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs



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

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


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

torch.set_grad_enabled(True)

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

for run in RunBuilder.get_runs(params):
    comment = f'-{run}'
#for lr, batch_size,shuffle in product(*param_values):
    lr = run.lr
    shuffle = run.shuffle
    batch_size = run.batch_size
    network = Network()
    prediction_loader = torch.utils.data.DataLoader(train_set,batch_size=10000)
    train_preds = get_all_preds(network,prediction_loader)
    print(train_preds.shape)

    sample = next(iter(train_set))

    image,label =sample#image is for one picture and images are for a batch of pictures


    pred = network(image.unsqueeze(0))#image shape nees to be (batch_size in_channels. height, width)

    optimizer = optim.Adam(network.parameters(),lr)#lr is learning rate
    data_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size = batch_size,
        shuffle = shuffle
    )
    sample = next(iter(data_loader))

    images,labels =sample#image is for one picture and images are for a batch of pictures


    pred = network(image.unsqueeze(0))#image shape nees to be (batch_size in_channels. height, width)
    #comment = f' batch_size={batch_size} lr = {lr}'
    tb = SummaryWriter(comment = comment)
    grid = torchvision.utils.make_grid(images)
    tb.add_image('images',grid)
    tb.add_graph(network,images)

    for epoch in range(5):
        total_loss = 0
        total_correct = 0
        for batch in data_loader:

            images,labels = batch

            preds = network(images)



            loss = F.cross_entropy(preds,labels)
            optimizer.zero_grad()
            
            loss.backward()
            


            

            optimizer.step()#update the weights of

            total_loss += loss.item()*batch_size
            total_correct += get_num_correct(preds,labels)
        tb.add_scalar('Loss',total_loss,epoch)
        tb.add_scalar('Number Correct',total_correct,epoch)
        tb.add_scalar('Accuracy',total_correct/len(train_set),epoch)
        
        # tb.add_histogram('conv1.bias',network.conv1.bias,epoch)
        # tb.add_histogram('conv1.weight',network.conv1.weight,epoch)
        # tb.add_histogram('conv1.weight.grad',network.conv1.weight.grad,epoch)
        for name, weight in network.named_parameters():
            tb.add_histogram(name,weight,epoch)
            tb.add_histogram(f'{name}.grad',weight.grad,epoch)
        print("epoch:",epoch,"accuracy:",total_correct/len(train_set))

    # with torch.no_grad():
    #     prediction_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
    #     train_preds = get_all_preds(network, prediction_loader)
    #     preds_correct = get_num_correct(train_preds,train_set.targets)
    #     print('accuracy:',preds_correct/len(train_set))
    #     stacked = torch.stack(
    #         (
    #             train_set.targets,
    #             train_preds.argmax(dim=1)
    #          )
    #          ,dim = 1
    #     )
    #     cmt = torch.zeros(10,10,dtype = torch.int32)
    #     for p in stacked:
    #         j,k = p.tolist()
    #         cmt[j,k] = cmt[j,k]+1
        
    #     cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
        # print(cm)
        
        # names = ('T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')
        # plt.figure(figsize=(10,10))
        # plot_confusion_matrix(cm,names)
        # while True
        #     plt.pause(10)
        
        # tb = SummaryWriter()

        # grid = torchvision.utils.make_grid(images)

        # tb.add_image('images',grid)
        # tb.add_graph(network,images)
tb.close()