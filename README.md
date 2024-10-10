## Introduction
Welcome to the repository for the study "Predicting the Urban Stormwater Drainage System State using the Graph-WaveNet," published in Sustainable Cities and Society. This repository contains the code used in our research to train a Graph-WaveNet model utilizing historical network state and rainfall data from SWMM simulations. For more details about our study, please refer to the paper available at https://doi.org/10.1016/j.scs.2024.105877

For the original PyTorch implementation of Graph-WaveNet, please refer to the paper Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019. (https://arxiv.org/abs/1906.00121)

## Requirements
- python 3.6
- see `requirements.txt`
- pytorch installation：
  1. download nvidia cuda12.0, or latest version (13 March 2023 update)  
  2. check driver version in CMD using 'nvidia-smi', high version will match cuda better
  3. download nvidia cudnn
  4. install torch(the following version works for my cuda), be careful, copy from https://pytorch.org/get-started/previous-versions/
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html

## Data Preparation
#### Generating training data from SWMM simulations
1. Prepare event-based SWMM output files
2. Run 'SWMM2TrainingData.ipynb' 

#### Generating .pkl file with adjacency matrix 
1. Extract node IDs and their distances from network data
2. Run 'gen_adj_mx.py'

## Training
```
python train.py
```
