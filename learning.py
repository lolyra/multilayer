import datasets
import encoders
from datasets import NSPLITS

import timm
import torch
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

import os
import numpy as np
import pickle as pk

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def fcon_output(dataname, modelname, resolution, path = ''):

    model = timm.create_model(modelname, pretrained=True, num_classes=0).eval().to(DEVICE)
    ds_data,_ = datasets.load(dataname, size=resolution, path=path, bsize=1, shuffle=False)

    print("Extracting feature vectors from fully-connected layer")
    n = len(ds_data)
    for i in range(n):
        filename = os.path.join(path,f"fcon/train_{i}.npz")
        if os.path.exists(filename):
            continue

        x = None
        y = None
        count=0
        total=len(ds_data[i]['train'].dataset)
        with torch.no_grad():
            for inputs,labels in ds_data[i]['train']:
                count+= inputs.shape[0]
                print("Round {0}/{3} - {1:04d}/{2:04d}".format(i+1,count,total,n),end='\r')

                output = model(inputs.to(DEVICE)).to('cpu').numpy()
                if x is None:
                    x = np.zeros((total,*output.shape[1:]))
                x[count-inputs.shape[0]:count] = output

                if y is None:
                    y = np.zeros((total,*labels.shape[1:]))
                y[count-inputs.shape[0]:count] = labels.numpy()
        np.savez_compressed(filename, x=x,y=y)
    print("{:<100}".format("Done"))


def conv_layers(dataloader, model, layers:int, s:str):
    x = {}
    y = None
    count = 0
    total = len(dataloader.dataset)
    with torch.no_grad():
        mdl = model.to(DEVICE)
        for inputs,labels in dataloader:
            count+= inputs.shape[0]
            print(s+"{0:04d}/{1:04d}".format(count,total),end='\r')

            if y is None:
                y = np.zeros((total,*labels.shape[1:]))
            y[count-inputs.shape[0]:count] = labels.numpy()

            output = mdl(inputs.to(DEVICE))
            for l in layers:
                aux = output[l].to('cpu').numpy()

                if l not in x:
                    x[l] = np.zeros((total,*aux.shape[1:]))
                x[l][count-inputs.shape[0]:count] = aux
        del mdl
    
    for l in layers:
        x[l] = x[l].reshape(x[l].shape[0],x[l].shape[1],-1).swapaxes(1,2)

    return x,y


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
        layer = layer.to(DEVICE)
        n = 500
        for i in range(0,x.shape[0],n):
            o = layer(x[i:i+n].to(DEVICE)).to('cpu').numpy()
            if i == 0:
                y = o
            else:
                y = np.append(y,o,axis=0)
            
    del layer
    return y


def process_autoencoder(x, layer:int, round:int, outsize:int, path:str):
    s = x.shape
    x = torch.tensor(x, dtype=torch.float)
    x = x.reshape(-1,s[2])
    n = 100

    filename = os.path.join(path,f"ae/{round}_{layer}.pkl")
    if os.path.exists(filename):
        ae = pk.load(open(filename,'rb')).to(DEVICE)
    else:
        ae = encoders.AutoEncoder(s[2], outsize).to(DEVICE)
        optimizer = torch.optim.Adam(ae.parameters())
        loss_fn = torch.nn.MSELoss()
        for epoch in range(20):
            for i in range(0,x.shape[0],n):
                z = x[i:i+n].to(DEVICE)
                y = ae(z)
                loss = loss_fn(y, z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        pk.dump(ae, open(filename,'wb'))

    with torch.no_grad():
        for i in range(0,x.shape[0],n):
            o = ae.encoder(x[i:i+n].to(DEVICE)).cpu().numpy()
            if i == 0:
                y = o
            else:
                y = np.append(y,o,axis=0)
    del ae
    y = y.reshape(s[0],-1,outsize)
    return y


def conv_output(dataname:str, modelname:str, resolution:int, layerlist:list[int], method:str='pca', path:str=''):

    model = timm.create_model(modelname, pretrained=True, features_only=True).eval()
    ds_data,_ = datasets.load(dataname, size=resolution, path=path, bsize=1, shuffle=False)

    print("Extracting local features from convolutional layers")
    n = len(ds_data)
    for i in range(n):
        s = f"Round {i+1}/{n} - "
        filename = os.path.join(path,f"conv/train_{i}.npz")
        if not os.path.exists(filename):
            x,y = conv_layers(ds_data[i]['train'],model,layerlist,s)

            xt = None
            ncomponents = 0
            for layer in layerlist:
                if ncomponents == 0:
                    ncomponents = x[layer].shape[2]
                    xt = x[layer]
                else:
                    if method == 'pca':
                        x[layer] = process_pca(x[layer],layer,i,ncomponents,path)
                    elif method in ['max','avg']:
                        x[layer] = process_layer(x[layer],ncomponents,method)
                    elif method == 'ae':
                        x[layer] = process_autoencoder(x[layer],layer,i,ncomponents,path)
                    else:
                        raise Exception("Invalid option")
                    xt = np.concatenate((xt,x[layer]),axis=1)
                del x[layer]
                
            np.savez_compressed(filename,x=xt,y=y)
            del xt
    print("{:<100}".format("Done"))


def fisher_vector(dataname, nkernels, path=''):
    n = NSPLITS.get(dataname,10)

    print("Pooling local features")
    for i in range(n):
        filename = os.path.join(path,f"fisher/train_{i}.npz")
        if not os.path.exists(filename):
            print(f"Round {i+1}/{n}")
            convfile = os.path.join(path,f"conv/train_{i}.npz")
            with np.load(convfile) as data:
                x = data['x']
                y = data['y']

            enc = encoders.EncoderGMM(nkernels)
            gmmfile = os.path.join(path,f"gmm/train_{i}.pkl")
            print("Learning GMM")
            if not os.path.exists(gmmfile):
                enc.fit(x)
                pk.dump(enc.gmm, open(gmmfile,"wb"))
            else:
                enc.gmm = pk.load(open(gmmfile,"rb"))
                enc.fitted = True
            
            v = []
            for j in range(0,len(x),1):
                print("Calculating Fisher Vectors {:04d}/{:04d}".format(j+1,len(x)),end='\r')
                v+= enc.transform(x[j:j+1]).tolist()
            del x
            v = np.array(v)
            np.savez_compressed(filename,x=v,y=y)
            print()
            del v
            del y
    print("{:<100}".format("Done"))


def normalize(x):
    x = np.sign(x)*np.fabs(x)**(0.5) # Power normalization
    x = x / np.linalg.norm(x, axis=1).reshape(-1,1) #L2 normalization
    x = np.sign(x)*np.sqrt(np.fabs(x)) # Bhattacharyya coefficient
    return x


def classifier(dataname:str, modelname:str, resolution:int, nkernels:int, layers:str, method:str, classifiername:str, path=''):
    from sklearn.svm import LinearSVC, SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.base import clone

    n = NSPLITS.get(dataname,10)

    print("Training classifier")
    for i in range(n):
        print(f"Round {i+1}/{n}",end="\r")
        with np.load(os.path.join(path,f"fisher/train_{i}.npz")) as data:
            x = normalize(data['x'])
            y = data['y']

        if classifiername == 'linear_svm':
            clf = LinearSVC(max_iter=5000,tol=0.001,dual=True,class_weight='balanced')
        elif classifiername == 'rbf_svm':
            clf = SVC(max_iter=5000, class_weight='balanced')
        elif classifiername == 'mlp':
            clf = MLPClassifier(random_state=0)
        elif classifiername == 'lda':
            clf = LinearDiscriminantAnalysis()
        else:
            raise Exception("Invalid classifier option")
        clf.fit(x,y)
        pk.dump(clf, open(os.path.join(path,f"classifier/{classifiername}_fv_{i}.pkl"),"wb"))

        with np.load(os.path.join(path,f"fcon/train_{i}.npz")) as data:
            z = normalize(data['x'])

        x = np.concatenate((z,x),axis=1)
        clf = clone(clf)
        clf.fit(x,y)
        pk.dump(clf, open(os.path.join(path,f"classifier/{classifiername}_fcfv_{i}.pkl"),"wb"))
    
    print("{:<100}".format("Done"))
