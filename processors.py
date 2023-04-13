import torch
import numpy as np

def fully_conn(dataloader, model, s:str):

    x = None
    y = None
    count=0
    total=len(dataloader.dataset)
    with torch.no_grad():
        mdl = model.to('cuda')
        for inputs,labels in dataloader:
            count+= inputs.shape[0]
            print(s+" FC  {0:04d}/{1:04d}".format(count,total),end='')

            output = mdl(inputs.to('cuda')).to('cpu').numpy()
            if x is None:
                x = output
            else:
                x = np.append(x,output,axis=0)

            if y is None:
                y = labels.numpy()
            else:
                y = np.append(y,labels.numpy(),axis=0)
        del mdl

    return x,y

def conv_layers(dataloader, model, l:int, s:str):

    x = None
    y = None
    count = 0
    total = len(dataloader.dataset)
    with torch.no_grad():
        for inputs,labels in dataloader:
            count+= inputs.shape[0]
            print(s+" CV{0} {1:04d}/{2:04d}".format(l,count,total),end='')
            output = model(inputs.to('cuda'))
            aux = output[l].to('cpu').numpy()
            del output

            if y is not None:
                y = np.append(y,labels.numpy(),axis=0)
            else:
                y = labels.numpy()

            
            if x is not None:
                x = np.append(x,aux,axis=0)
            else:
                x = aux

    return x.reshape(x.shape[0],x.shape[1],-1).swapaxes(1,2),y
