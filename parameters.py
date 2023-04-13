import processors
import datasets
import encoders

import timm
import torch
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

import os
import numpy as np
import pickle as pk

def make_dir(path:str=''):

    for directory in ['conv','fcon','fisher','gmm','pca','results']:
        os.makedirs(os.path.join(path,directory),exist_ok=True)

def clean_dir(listdir, path:str=''):

     for directory in listdir:
        filepath = os.path.join(path,directory)
        if os.path.exists(filepath):
            for filename in os.listdir(filepath):
                os.remove(os.path.join(filepath,filename))


def fcon_output(dataname, modelname, resolution, path = ''):

    model = timm.create_model(modelname, pretrained=True, num_classes=0).eval()
    ds_data,_ = datasets.load(dataname, size=resolution, path=path, bsize=20, shuffle=False)

    for i in range(len(ds_data)):
        for stage in ['train','val']:
            s = "\r{:02d}/{:<5}".format(i,stage)
            filename = os.path.join(path,f"fcon/{stage}_{i}.npz")
            if not os.path.exists(filename):
                x,y = processors.fully_conn(ds_data[i][stage], model, s)
                np.savez_compressed(filename, x=x,y=y)

def process_pca(x, layer:int, round:int, ncomponents:int, path:str=''):

    s = x.shape
    if s[2] == ncomponents:
        return x

    x = x.reshape(-1,s[2])
    filename = os.path.join(path,f'pca/{round}_{layer}.pkl')
    if os.path.exists(filename):
        pca = pk.load(open(filename,'rb'))
    else:
        pca = PCA(ncomponents,random_state=0)
        pca.fit(x)
        pk.dump(pca, open(filename,"wb"))

    x = pca.transform(x).reshape(s[0],-1,ncomponents)
    
    return x

def process_layer(x,outsize:int,method:str):

    x = torch.tensor(x)

    insize = x.shape[2]
    stride = insize//outsize
    kernel = insize-stride*outsize+1

    if method == 'max':
        layer = torch.nn.MaxPool2d((1,kernel),(1,stride),0)
    else:
        layer = torch.nn.AvgPool2d((1,kernel),(1,stride),0)
    
    with torch.no_grad():
        layer = layer.to('cuda')
        n = 500
        for i in range(0,x.shape[0],n):
            o = layer(x[i:i+n].to('cuda')).to('cpu').numpy()
            if i == 0:
                y = o
            else:
                y = np.append(y,o,axis=0)
            
    del layer
    return y

def conv_output(dataname:str, modelname:str, resolution:int, layerlist:list[int], method:str='pca', path:str=''):

    model = timm.create_model(modelname, pretrained=True, features_only=True).eval().to('cuda')
    ds_data,_ = datasets.load(dataname, size=resolution, path=path, bsize=20, shuffle=False)

    for i in range(len(ds_data)):
        for stage in ['train','val']: 
            s = "\r{:02d}/{:<5}".format(i,stage)
            filename = os.path.join(path,f"conv/{stage}_{i}.npz")
            if not os.path.exists(filename):
                xt = None
                ncomponents = 0
                for layer in layerlist:
                    x,y = processors.conv_layers(ds_data[i][stage],model,layer,s)

                    if ncomponents == 0:
                        ncomponents = x.shape[2]
                        xt = x
                    else:
                        if method == 'pca':
                            x = process_pca(x,layer,i,ncomponents,path)
                        elif method in ['max','avg']:
                            x = process_layer(x,ncomponents,method)
                        else:
                            raise Exception("Invalid option")
                        xt = np.concatenate((xt,x),axis=1)
                    
                np.savez_compressed(filename,x=xt,y=y)
                del xt    
    del model


def fisher_output(dataname, nkernels, path=''):
    n = 4 if dataname == 'kth' else 10

    for i in range(n):
        for stage in ['train','val']:
            s = "\r{:02d}/{:<5}".format(i,stage)
            filename = os.path.join(path,f"fisher/{stage}_{i}.npz")
            if not os.path.exists(filename):
                convfile = os.path.join(path,f"conv/{stage}_{i}.npz")
                with np.load(convfile) as data:
                    x = data['x']
                    y = data['y']

                enc = encoders.EncoderGMM(nkernels)
                gmmfile = os.path.join(path,f"gmm/train_{i}.pkl")
                print(s+" GMM {:<9}".format(""),end='')
                if not os.path.exists(gmmfile):
                    enc.fit(x)
                    pk.dump(enc.gmm, open(gmmfile,"wb"))
                else:
                    enc.gmm = pk.load(open(gmmfile,"rb"))
                    enc.fitted = True
                
                v = []
                for j in range(0,len(x),20):
                    print(s+" FV  {:04d}/{:04d}".format(j+20,len(x)),end='')
                    v+= enc.transform(x[j:j+20]).tolist()
                del x
                v = np.array(v)
                np.savez_compressed(filename,x=v,y=y)
                del v
                del y

def classify(dataname:str, modelname:str, resolution:int, nkernels:int, layers:str, method:str, path=''):
    n = 4 if dataname == 'kth' else 10

    acc1 = []
    acc2 = []
    acc3 = []
    y_true = []
    y_pre1 = []
    y_pre2 = []
    y_pre3 = []

    for i in range(n):   
        fc = LinearSVC(max_iter=5000,tol=0.001)
        fv = LinearSVC(max_iter=5000,tol=0.001)
        fcfv = LinearSVC(max_iter=5000,tol=0.001)

        with np.load(os.path.join(path,f"fcon/train_{i}.npz")) as data:
            y = data['y']
            x = data['x']
        x = np.sign(x)*np.fabs(x)**(0.5) # Power normalization
        x = x / np.linalg.norm(x, axis=1).reshape(-1,1) #L2 normalization
        x = np.sign(x)*np.sqrt(np.fabs(x)) # Bhattacharyya coefficient
        fc.fit(x,y)

        with np.load(os.path.join(path,f"fisher/train_{i}.npz")) as data:
            z = data['x']
        z = np.sign(z)*np.fabs(z)**(0.5)  # Power normalization
        z = z / np.linalg.norm(z, axis=1).reshape(-1,1) #L2 normalization
        z = np.sign(z)*np.sqrt(np.fabs(z)) # Bhattacharyya coefficient
        fv.fit(z,y)

        x = np.concatenate((x,z),axis=1)
        fcfv.fit(x,y)

        with np.load(os.path.join(path,f"fcon/val_{i}.npz")) as data:
            y = data['y']
            x = data['x']
        x = np.sign(x)*np.fabs(x)**(0.5) # Power normalization
        x = x / np.linalg.norm(x, axis=1).reshape(-1,1) #L2 normalization
        x = np.sign(x)*np.sqrt(np.fabs(x)) # Bhattacharyya coefficient
        acc1.append(fc.score(x,y))
        y_pre1+= fc.predict(x).tolist()

        with np.load(os.path.join(path,f"fisher/val_{i}.npz")) as data:
            z = data['x']
        z = np.sign(z)*np.fabs(z)**(0.5) # Power normalization
        z = z / np.linalg.norm(z, axis=1).reshape(-1,1) #L2 normalization
        z = np.sign(z)*np.sqrt(np.fabs(z)) # Bhattacharyya coefficient
        acc2.append(fv.score(z,y))
        y_pre2+= fv.predict(z).tolist()

        x = np.concatenate((x,z),axis=1)
        acc3.append(fcfv.score(x,y))
        y_pre3+= fcfv.predict(x).tolist()

        #acc3.append(sum((fc.decision_function(x)+fv.decision_function(z)).argmax(axis=1)==y)/len(y))
        #y_pre3+= (fc.decision_function(x)+fv.decision_function(z)).argmax(axis=1).tolist()
        y_true+= y.tolist()
    
    m = confusion_matrix(y_true, y_pre1)
    np.savez_compressed(os.path.join(path,f"results/mat_{dataname}_{modelname}_{nkernels}_{resolution}_{layers}_{method}_fc.npz"),m=m)
    m = confusion_matrix(y_true, y_pre2)
    np.savez_compressed(os.path.join(path,f"results/mat_{dataname}_{modelname}_{nkernels}_{resolution}_{layers}_{method}_fv.npz"),m=m)
    m = confusion_matrix(y_true, y_pre3)
    np.savez_compressed(os.path.join(path,f"results/mat_{dataname}_{modelname}_{nkernels}_{resolution}_{layers}_{method}_fcfv.npz"),m=m)

    print("\r{} {} {} {} {} {}".format(dataname, modelname, nkernels, resolution, layers, method))
    print("FC = {:.4f} FV = {:.4f} FC + FV = {:.4f}".format(np.mean(acc1),np.mean(acc2),np.mean(acc3)))
    np.savez_compressed(os.path.join(path,f"results/acc_{dataname}_{modelname}_{nkernels}_{resolution}_{layers}_{method}.npz"),fc=acc1, fv=acc2, fcfv=acc3)
