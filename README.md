# Multilayer deep feature extraction for visual texture recognition

Convolutional neural networks have shown successful results in image classification achieving real-time results superior to the human level. However, texture images still pose some challenge to these models due, for example, to the limited availability of data for training in several problems where these images appear, high inter-class similarity, the absence of a global viewpoint of the object represented, and others. In this context, the present paper is focused on improving the accuracy of convolutional neural networks in texture classification. This is done by extracting features from multiple convolutional layers of a pretrained neural network and aggregating such features using Fisher vector. The reason for using features from earlier convolutional layers is obtaining information that is less domain specific. We verify the effectiveness of our method on texture classification of benchmark datasets, as well as on a practical task of Brazilian plant species identification. In both scenarios, Fisher vectors calculated on multiple layers outperform state-of-art methods, confirming that early convolutional layers provide important information about the texture image for classification.

https://arxiv.org/abs/2208.10044

```
@misc{lyra2022multilayer,
      title={Multilayer deep feature extraction for visual texture recognition}, 
      author={Lucas O. Lyra and Antonio Elias Fabris and Joao B. Florindo},
      year={2022},
      eprint={2208.10044},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Preparation

In order to run the code, proceed to install the required packages. 
Before installing, we advise creating a virtual environment.

`$ python3 -m venv <path_to_virutal_env> --upgrade-deps`

After creating the virtual environment, activate it and install the required packages.

`$ source <path to virtual environment>/bin/activate`

`$ pip install -r requirements.txt`

## Execution

Open _main.py_ and specify a directory to read/write data. Also specify other desired configurations for the run.

Navigate to the specied directory and create a directory named _databases_:

`$ cd <path to directory to read/write>`

`$ mkdir databases`

Download the desired database and extract it to _databases_ directory.
The following lines are an example for KTH-TIPS2-b:

`$ wget https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b_col_200x200.tar`

`$ tar -xf kth-tips2-b_col_200x200.tar -C databases/`

`$ rm kth-tips2-b_col_200x200.tar`

After downloading and extracting, run the code 

`$ python3 <path to main.py>`

The results will be saved to a directory _results_ inside the specied directory to read/write.
