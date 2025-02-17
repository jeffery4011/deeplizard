import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple
from collections import OrderedDict

sequential = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    ，nn.BatchNorm2d(6)
    , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.Flatten(start_dim=1)  
    , nn.Linear(in_features=12*4*4, out_features=120)
    , nn.ReLU()
    , nn.BatchNorm1d(120)
    , nn.Linear(in_features=120, out_features=60)
    , nn.ReLU()
    , nn.Linear(in_features=60, out_features=10)
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

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run',params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


params = OrderedDict(
    lr = [.01]
    ,batch_size =[1000,10000,20000]
    ,num_workers =[0,1]#,2,4,8,16]
    ,device =['cuda','cpu']
    #,shuffle=[True,False]
    ,trainset =['not_normal','normal']
)


class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data =[]
        self.run_start_time =None

        self.network = None
        self.loader = None
        self.tb =None

    def begin_run(self,run,network,loader):
    
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images.to(getattr(run,'device','cpu')))

    def end_run(self):
        self.tb.close()
        self. epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss =0
        self.epoch_num_correct = 0
    
    def get_num_correct(self,preds,labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

    #clear_output(wait = True)
    #display(df)

    def track_loss(self,loss):
        self.epoch_loss +=loss.item()*self.loader.batch_size
    
    def track_num_correct(self,preds,labels):
        self.epoch_num_correct += self.get_num_correct(preds,labels)
    
    def save(self,fileName):

        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

if __name__ ==  '__main__':
    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST'
        ,train =True
        ,download=True
        ,transform=transforms.Compose(
            [
                transforms.ToTensor()
                #normalize
                
            ]
        )
    )
    m = RunManager()
    loader = DataLoader(train_set,batch_size=len(train_set),num_workers = 1)
    data = next(iter(loader))
    mean = data[0].mean()
    std = data[0].std()
    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST'
        ,train =True
        ,download=True
        ,transform=transforms.Compose(
            [
                transforms.ToTensor()
                #normalize
                
            ]
        )
    )
    train_set_normal = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST'
        ,train =True
        ,download=True
        ,transform=transforms.Compose(
            [
                transforms.ToTensor()
                #normalize
                ,transforms.Normalize(mean,std)
            ]
        )
    )
    trainset={
    'not_normal':train_set
    ,'normal':train_set_normal
    }
    for run in RunBuilder.get_runs(params):
        device = torch.device(run.device)
        network = Network().to(device)#sequential().to(device)
        loader = DataLoader(train_set,batch_size=run.batch_size,num_workers = run.num_workers)
        data = next(iter(loader))
        mean = data[0].mean()
        std = data[0].std()
        optimizer = optim.Adam(network.parameters(),lr=run.lr)

        m.begin_run(run,network,loader)
        for epoch in range(5):
            m.begin_epoch()
            for batch in loader:

                images =batch[0].to(device) 
                labels =batch[1].to(device)
                preds = network(images)
                loss = F.cross_entropy(preds,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                m.track_loss(loss)
                m.track_num_correct(preds,labels)
            m.end_epoch()
        m.end_run()
    m.save('results')