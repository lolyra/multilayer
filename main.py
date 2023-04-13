import parameters

def main(datasets, modelnames, resolutions, kernels, layers, methods, path):

    parameters.make_dir(path)
    for dataname in datasets:
        for modelname in modelnames:
            for resolution in resolutions:
                parameters.fcon_output(dataname, modelname, resolution, path)
                for layer in layers:
                    for method in methods:
                        parameters.conv_output(dataname, modelname, resolution, layer, method, path)
                        for kernel in kernels:
                            parameters.fisher_output(dataname, kernel, path)
                            parameters.classify(dataname, modelname, resolution, kernel, ''.join([str(l) for l in layer]), method, path)
                            parameters.clean_dir(['fisher','gmm'],path)
                        parameters.clean_dir(['conv','pca'],path)
                        parameters.clean_dir(['conv'],path)
                parameters.clean_dir(['fcon'],path)

if __name__ == "__main__":
    main(['dtd'], # Databases
        ['tf_efficientnet_b5'], # CNN Models
        [320], # Input image resolution
        [16], # Number of Gaussian distributions
        [[3,4]], # Number of CNN layers
        ['pca','avg','max'], # List of methods for combining layers
        '/var/tmp/lolyra/data') # Path to folder where data will be read and saved

