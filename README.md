# Towards Privacy-Preserving and Domain-Aware Knowledge Graph Entity Representations

This repository has the code for the project in course CS6216. The code is built on the original repository titled [BLP](https://github.com/dfdazac/blp), which heavily inspired our project. The torchcoder directory in this repository is the repository titled [TorchCoder](https://github.com/hellojinwoo/TorchCoder). 

## Create new conda env:
```
conda create --name cs6216 python=3.7
conda activate cs6216
pip install -r requirements.txt
mkdir output
```
## Running the original baseline DKRL-BERT
```
chmod +x baseline.sh
./baseline.sh
```
## Running the experiments for KG entity representation
```
python bert_dkrl.py
python flair_dkrl.py
python aec.py
```
To change the number of epochs, go to the last line in each of these files where the link_prediction function is called, and modify the parameters there. 
