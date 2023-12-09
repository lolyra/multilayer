import learning
import evaluating
import os

def make_dir(path:str=''):
    for directory in ['conv','fcon','fisher','gmm','pca','ae','classifier','results']:
        os.makedirs(os.path.join(path,directory),exist_ok=True)

def clean_dir(listdir, path:str=''):
     for directory in listdir:
        filepath = os.path.join(path,directory)
        if os.path.exists(filepath):
            for filename in os.listdir(filepath):
                os.remove(os.path.join(filepath,filename))

def main(datasets, modelnames, resolutions, kernels, layers, methods, classifiers, snrs, path):
    make_dir(path)
    for dataname in datasets:
        for modelname in modelnames:
            for resolution in resolutions:
                learning.fcon_output(dataname, modelname, resolution, path)
                for layer in layers:
                    for method in methods:
                        learning.conv_output(dataname, modelname, resolution, layer, method, path)
                        for kernel in kernels:
                            learning.fisher_vector(dataname, kernel, path)
                            for classifiername in classifiers:
                                learning.classifier(dataname, modelname, resolution, kernel, ''.join([str(l) for l in layer]), method, classifiername,  path)
                                for snr in snrs:
                                    evaluating.classify(dataname, modelname, resolution, kernel, layer, method, classifiername, path, snr)
                            clean_dir(['fisher','gmm','classifier'],path)
                        clean_dir(['conv','pca'],path)
                clean_dir(['fcon'],path)

if __name__ == "__main__":
    p = os.path.join(os.environ['HOME'],'data')

    main(['fmd'], # Dataset (options: '1200Tex','kth','fmd','dtd','gtos','umd','uiuc')
         ['tf_efficientnet_b5'], # CNN backbone (names must match models available in Pytorch Image Models library)
         [320], # Image width
         [16], # Number of kernels
         [[3,4]], # List of indexes of feature maps to be used (in EfficientNet 4 is the output of block 6)
         ['ae'], # Method for dimensionality reduction (options: 'pca','avg','max','ae')
         ['linear_svm'], # Classifier
         [None], # Signal to Noise Ratio (None means no noise is applied)
         p # Path to load data from and save data to
         )
   
