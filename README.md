# MultiFisherNet: a multilevel approach to deep filter banks in texture recognition

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
