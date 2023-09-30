import os, random
import numpy as np

from torch import manual_seed, randn_like
from torch.utils.data import Subset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

NSPLITS = {'kth':4,'gtos':5}

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

def load_gtos(path, datatransform, bsize, shuffle):
    datadir = os.path.join(path,'gtos')
    dataset = ImageFolder(os.path.join(datadir,'images'), datatransform)
        
    dataloader = []
    for i in range(NSPLITS.get('gtos',10)):
        with open(os.path.join(datadir,'labels/train{}.txt'.format(i+1)),'r') as f:
            data = []
            for x in f.readlines():
                p = x.split(' ')[0]
                imgs = random.sample(os.listdir(os.path.join(datadir,'images',p)),2)
                for img in imgs:
                    data.append(os.path.join(p,img))
        idx_train = [x for x in range(len(dataset)) if dataset.imgs[x][0].split('images/')[-1] in data]
        
        with open(os.path.join(datadir,'labels/test{}.txt'.format(i+1)),'r') as f:
            data = [x.split(' ')[0].split('/')[1] for x in f.readlines()]
        idx_test = [x for x in range(len(dataset)) if dataset.imgs[x][0].split('/')[-2] in data]
        
        dataloader.append({'train': DataLoader(Subset(dataset, idx_train),bsize,shuffle),
        'val': DataLoader(Subset(dataset, idx_test),bsize,shuffle)})

    return dataloader, dataset.classes

def load_dtd(path, datatransform, bsize, shuffle):
    datadir = os.path.join(path,'dtd')
    dataset = ImageFolder(os.path.join(datadir,'images'), datatransform)
        
    dataloader = []
    for i in range(NSPLITS.get('dtd',10)):
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

def load_other(path, dataname, datatransform, bsize, shuffle, ratio = 0.5):
    datadir = os.path.join(path,dataname)
    dataset = ImageFolder(datadir, datatransform)

    n = int(ratio * len(dataset))
    d = len(dataset) - n

    dataloader = []
    for i in range(NSPLITS.get(dataname,10)):
        train_data, test_data = random_split(dataset,[n,d])

        dataloader.append({'train': DataLoader(train_data,bsize,shuffle),
        'val': DataLoader(test_data,bsize,shuffle)})
    
    return dataloader, dataset.classes


def load(dataset, size:int , path:str='', bsize:int=1, shuffle:bool=True, snr:float=None):

    if dataset in ['dtd','kth']:
        size = (size,size)

    datadir = os.path.join(path,"databases")

    def gaussian_noise(img):
        if snr is None:
            return img 
        sigma = 0.5/snr
        return img + sigma * randn_like(img)

    data_transforms = transforms.Compose([
                      transforms.Resize(size,interpolation=transforms.InterpolationMode.BICUBIC)
                      ,transforms.ToTensor()
                      ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ,gaussian_noise
                      ])
    # Reproductibility
    np.random.seed(0)
    random.seed(0)
    manual_seed(0)

    if dataset == 'kth':
        return load_kth(datadir, data_transforms, bsize, shuffle)

    if dataset == 'dtd':
        return load_dtd(datadir, data_transforms, bsize, shuffle)
    
    if dataset == 'gtos':
        return load_gtos(datadir, data_transforms, bsize, shuffle)
    
    if dataset == 'fmd':
        return load_other(datadir, 'fmd/image', data_transforms, bsize, shuffle, 0.5)

    return load_other(datadir, dataset, data_transforms, bsize, shuffle, 0.5)
    
