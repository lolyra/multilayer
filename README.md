# Multilayer Deep Feature Extraction for Visual Texture Recognition

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
