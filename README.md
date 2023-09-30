# Multilayer Deep Feature Extraction for Visual Texture Recognition

Convolutional neural networks have shown successful results in image classification achieving real-time results superior to the human level. However, texture images still pose some challenge to these models due, for example, to the limited availability of data for training in several problems where these images appear, high inter-class similarity, the absence of a global viewpoint of the object represented, and others. In this context, the present paper is focused on improving the accuracy of convolutional neural networks in texture classification. This is done by extracting features from multiple convolutional layers of a pretrained neural network and aggregating such features using Fisher vector. The reason for using features from earlier convolutional layers is obtaining information that is less domain specific. We verify the effectiveness of our method on texture classification of benchmark datasets, as well as on a practical task of Brazilian plant species identification. In both scenarios, Fisher vectors calculated on multiple layers outperform state-of-art methods, confirming that early convolutional layers provide important information about the texture image for classification.

https://doi.org/10.48550/arXiv.2208.10044

## Preparation

In order to run the code, proceed to install the required packages. 
Before installing, we advise [creating a virtual environment](https://docs.python.org/3/library/venv.html).

After creating the virtual environment, activate it and install the required packages.

`$ pip install -r requirements.txt`

## Execution

Training and evaluation is performed in `main.py`. 
In order to specify desired configurations for the experiment and path to read/write data, open the file and navigate to the end of it.
Instructions for changing parameters are commented.
 
Before running, you need to download and extract the desired database.
Navigate to the specied directory and create a directory named _databases_:

`$ cd <path to directory to read/write>`

`$ mkdir databases`

Download the desired database and extract it to _databases_ directory.
The following lines are an example for KTH-TIPS2-b:

`$ wget https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b_col_200x200.tar`

`$ tar -xf kth-tips2-b_col_200x200.tar -C databases`

`$ rm kth-tips2-b_col_200x200.tar`

After downloading and extracting, run the code 

`$ python <path to main.py>`

The results will be saved to a directory _results_ inside the specied directory to read/write.

In order to plot results, open file `plot_results.py` and navigate to the end of the file.
There, call funcion _main_ with the parameters of the desired experiment.
