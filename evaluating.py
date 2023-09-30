import datasets
from learning import normalize, DEVICE
from encoders import EncoderGMM
import timm
import torch
import os
import numpy as np
import pickle as pk

def add_noise(x):
    return x

def classify(dataname, modelname, resolution, kernels, layers, method, path, snr:float=None):

    ds_data,_ = datasets.load(dataname, size=resolution, path=path, bsize=1, shuffle=False, snr=snr)
    model_fc = timm.create_model(modelname, pretrained=True, num_classes=0).eval().to(DEVICE)
    model_fv = timm.create_model(modelname, pretrained=True, features_only=True).eval().to(DEVICE)
    torch.set_grad_enabled(False)

    layers_sizes = []
    x = torch.randn((1,3,112,112)).to(DEVICE)
    o = model_fv(x)
    del x
    for y in o:
        layers_sizes.append(y.shape[1])
    outsize = layers_sizes[layers[0]]
        
    if method == 'pca':
        def reduce_dim(x, mdl):
            x = x.numpy()
            s = x.shape
            x = x.reshape(-1,s[2])
            return mdl(x).reshape(s[0],-1,outsize)
    else:
        def reduce_dim(x, mdl):
            return mdl(x).numpy()

    if snr is None:
        print("Evaluating model")
    else:
        print(f"Evaluating model with SNR {snr}")
   
    n = len(ds_data)
    for i in range(n):
        enc = EncoderGMM(kernels)
        gmmfile = os.path.join(path,f"gmm/train_{i}.pkl")
        pcafile = os.path.join(path,f"pca/{i}_{{}}")
        enc.gmm = pk.load(open(gmmfile,'rb'))
        enc.fitted = True

        mdls = {layers[0]:lambda x: x}
        pcas = {}
        for l in layers[1:]:
            if method == 'pca':
                pcafile = os.path.join(path,f"pca/{i}_{l}.pkl")
                pcas[l] = pk.load(open(pcafile,'rb'))
                mdls[l] = lambda x: pcas[l].transform(x)
                continue
            insize = layers_sizes[l]
            stride = insize//outsize
            kernel = insize-stride*outsize+1
            if method == 'max':
                mdls[l] = torch.nn.MaxPool2d((1,kernel),(1,stride),0)
            else:
                mdls[l] = torch.nn.AvgPool2d((1,kernel),(1,stride),0)

        svm_fv = pk.load(open(os.path.join(path,f"svm/fv_{i}.pkl"),'rb'))
        svm_fcfv = pk.load(open(os.path.join(path,f"svm/fcfv_{i}.pkl"),'rb'))

        total = len(ds_data[i]['val'].dataset)
        y_true = np.zeros(total)
        y_fv = np.zeros(total)
        y_fcfv = np.zeros(total)
        count = 0
        for inputs,labels in ds_data[i]['val']:
            count+= inputs.shape[0]
            print("Round {}/{} - {:05d}/{:05d}".format(i+1,n,count,total),end="\r")
            inputs = add_noise(inputs).to(DEVICE)
            x_fc = model_fc(inputs).to('cpu').numpy()
            o = model_fv(inputs)
            
            x_fv = None
            for l in layers:
                s = o[l].shape
                x = o[l].to('cpu').reshape(s[0],s[1],-1).swapaxes(1,2)
                x = reduce_dim(x,mdls[l])
                if x_fv is None:
                    x_fv = x
                else:
                    x_fv = np.concatenate((x_fv,x),axis=1)

            x_fc = normalize(x_fc)
            x_fv = normalize(enc.transform(x_fv))
            x = np.concatenate((x_fc,x_fv),axis=1)

            y_fv[count-1] = svm_fv.predict(x_fv)[0]
            y_fcfv[count-1] = svm_fcfv.predict(x)[0]
            y_true[count-1] = labels.numpy()[0]

        layer = ''.join([str(l) for l in layers])
        endfile = '.npz' if snr is None else f'_{snr}.npz'
        np.savez_compressed(os.path.join(path,f"results/{dataname}_{modelname}_{kernels}_{resolution}_{layer}_{method}_{i}{endfile}"),
                            y_pred1 = y_fv, 
                            y_pred2 = y_fcfv,
                            y_true = y_true)

    print("{:<100}".format("Done"))

