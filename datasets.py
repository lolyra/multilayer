import os
import numpy as np

from torch import manual_seed
from torch.utils.data import Subset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

def load_kth(path, datatransform, bsize, shuffle):
    datadir = os.path.join(path,'KTH-TIPS2-b')
    dataset = ImageFolder(datadir, datatransform)

    dataloader = []
    samplelist = ['sample_a','sample_b','sample_c','sample_d']

    for sample in samplelist:
        idx_train = [i for i in range(len(dataset)) if dataset.imgs[i][0].split('/')[-2] != sample]
        idx_test = [i for i in range(len(dataset)) if dataset.imgs[i][0].split('/')[-2] == sample]
    
        dataloader.append({'train': DataLoader(Subset(dataset, idx_train),bsize,shuffle),
        'val': DataLoader(Subset(dataset, idx_test),bsize,shuffle)})

    return dataloader, dataset.classes

def load_dtd(path, datatransform, bsize, shuffle):
    datadir = os.path.join(path,'dtd')
    dataset = ImageFolder(os.path.join(datadir,'images'), datatransform)
        
    dataloader = []
    for i in range(10):
        with open(os.path.join(datadir,'labels/train{}.txt'.format(i+1)),'r') as f:
            data = [x.strip() for x in f.readlines()]
        with open(os.path.join(datadir,'labels/val{}.txt'.format(i+1)),'r') as f:
            data+= [x.strip() for x in f.readlines()]
        idx_train = [x for x in range(len(dataset)) if dataset.imgs[x][0].split('images/')[-1] in data]
        
        with open(os.path.join(datadir,'labels/test{}.txt'.format(i+1)),'r') as f:
            data = [x.strip() for x in f.readlines()]
        idx_test = [x for x in range(len(dataset)) if dataset.imgs[x][0].split('images/')[-1] in data]
        
        dataloader.append({'train': DataLoader(Subset(dataset, idx_train),bsize,shuffle),
        'val': DataLoader(Subset(dataset, idx_test),bsize,shuffle)})

    return dataloader, dataset.classes

def load_other(path, dataname, datatransform, bsize, shuffle):
    datadir = os.path.join(path,dataname)
    dataset = ImageFolder(datadir, datatransform)

    n = len(dataset)//2
    d = len(dataset)%2

    dataloader = []
    for i in range(10):
        train_data, test_data = random_split(dataset,[n,n+d])

        dataloader.append({'train': DataLoader(train_data,bsize,shuffle),
        'val': DataLoader(test_data,bsize,shuffle)})
    
    return dataloader, dataset.classes


def load(dataset, size:int , path:str='', bsize:int=40, shuffle:bool=True):

    if dataset in ['dtd','kth']:
        size = (size,size)

    datadir = os.path.join(path,"databases")

    data_transforms = transforms.Compose([
                      transforms.Resize(size,interpolation=transforms.InterpolationMode.BICUBIC)
                      ,transforms.ToTensor()
                      ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])
    # Reproductibility
    np.random.seed(0)
    manual_seed(0)

    if dataset == 'kth':
        return load_kth(datadir, data_transforms, bsize, shuffle)

    if dataset == 'dtd':
        return load_dtd(datadir, data_transforms, bsize, shuffle)
    
    if dataset == 'fmd':
        return load_other(datadir, 'fmd/image', data_transforms, bsize, shuffle)
    
    return load_other(datadir, dataset, data_transforms, bsize, shuffle)
    
