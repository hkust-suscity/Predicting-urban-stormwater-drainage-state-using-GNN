# GWN for urban stormwater drainage system state prediction

The original pytorch implementation of Graph WaveNet is from the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121).

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
#### SWMM simulation and generating training data (Google Colab)
1. Extract rainfall events from time series 
2. Prepare climate data(e.g, daily), include maximum and minimum temperature, wind speed
3. Create an .inp file in SWMM software with network data, 3 year climate data as a baseline, and change only rainfall in subsequent simulations
4. Run '4_events-SWMM-dataset.ipynb' with extracted rainfall events to generate training data for GWN

#### Generating .pkl file with adjacency matrix 
1. Extract node IDs and their distances from network data
2. Run 'gen_adj_mx.py'

## Training
```
python train.py
```
