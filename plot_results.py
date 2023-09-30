import numpy
import os
from sklearn.metrics import (
    precision_score, recall_score,
    accuracy_score, f1_score, confusion_matrix
)
from matplotlib import pyplot

from datasets import NSPLITS

def plot_matrix(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    n_labels = max(y_true)+1
    step = n_labels//12+1
    
    labels = numpy.arange(0,n_labels,step).astype(int)
    pyplot.rcParams.update({'font.size': 12})
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(m, cmap='Greys')
    fig.colorbar(cax)
    ax.set_xticks(labels)
    ax.set_xticklabels(labels+1)
    ax.set_yticks(labels)
    ax.set_yticklabels(labels+1)
    
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.xaxis.set_ticks_position('bottom')
    
    pyplot.show()

def main(database, modelname, resolution, kernels, layer_list, pooling, snr, path):
    layers = ''.join([str(x) for x in layer_list])
    endfile = '.npz' if snr is None else f'_{snr}.npz'
    n = NSPLITS.get(database,10)
    acc = numpy.zeros((2,4,n))
    y_true = []
    y_pred = []
    for i in range(n):
        filename = os.path.join(path, f"{database}_{modelname}_{kernels}_{resolution}_{layers}_{pooling}_{i}{endfile}")
        data = numpy.load(filename)
        y0 = data['y_true']
        y1 = data['y_pred1']
        y2 = data['y_pred2']
        acc[0,0,i] = accuracy_score(y0,y1)
        acc[1,0,i] = accuracy_score(y0,y2)
        acc[0,1,i] = precision_score(y0,y1,average='macro')
        acc[1,1,i] = precision_score(y0,y2,average='macro')
        acc[0,2,i] = recall_score(y0,y1,average='macro')
        acc[1,2,i] = recall_score(y0,y2,average='macro')
        acc[0,3,i] = f1_score(y0,y1,average='macro')
        acc[1,3,i] = f1_score(y0,y2,average='macro')
        y_true+= y0.tolist()
        y_pred+= y2.tolist()

    ft = ['FV','FV+FC']
    print("{:<14} | {:<11} | {:<11} | {:<11} | {:<11}".format("Feature Vector", "Accuracy", "Precision", "Recall", "F1-Score")) 
    for i in range(acc.shape[0]):
        print("{:<14}".format(ft[i]),end=" ")
        for j in range(acc.shape[1]):
            print("| {:.1f} +- {:.1f}".format(100*acc[i,j,:].mean(),196*acc[i,j,:].std()/numpy.sqrt(n)),end=" ")
        print()

    plot_matrix(y_true, y_pred)
    

if __name__ == "__main__":
    main('kth', # Dataset
         'tf_efficientnet_b5', # CNN backbone
         320, # Image width
         16, # Number of kernels
         [3,4], # Indexes of feature maps used
         'pca', # Diimensionality redution method
         60, # SNR
         '/var/tmp/lolyra/data/results' # Path to results
        )
