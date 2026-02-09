# 5_weightCNN.py
# Author: Xiaoqian Sun, 10/03/2025
# Fucntion: Train network on pooled (raw/scaled) network ccgs and predict on another unseen network. 
#           Try to see if MCC can be improved if trained on networks cross domains


# Import Packages
#=================================================================================================================
import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from matplotlib import cm, colors
from scipy.stats import norm
import scipy.special as special
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.manifold import TSNE
from scipy.stats import spearmanr
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader


# simulation package
import sys 
import importlib
sys.path.insert(0, '.../PointNeuron_Simulation') # modify path
import utils, signals, connectivity, neuron, visualization, simulation
importlib.reload(utils)
importlib.reload(neuron)
importlib.reload(signals)
importlib.reload(connectivity)
importlib.reload(visualization)
importlib.reload(simulation)
from utils import *; from visualization import *

import Functions
importlib.reload(Functions)
from Functions import *

import warnings
warnings.filterwarnings('ignore')


# argumetns
#=================================================================================================================
parser = argparse.ArgumentParser("Traning WeightCNN")

parser.add_argument('--dataloaderType', default='Raw', type=str)          # Raw/Scale/Dual
parser.add_argument('--dataloaderName', default='networkLoader_2_TTV.pkl', type=str)
parser.add_argument('--resultFolderName', default='network_2', type=str)

parser.add_argument('--unseen_dlName', default='networkLoader1.pkl', type=str)  #networkLoader_scale.pkl

parser.add_argument('--numRuns', default=30, type=int)
parser.add_argument('--trainingEpochs', default=120, type=int)
parser.add_argument('--trainingPatience', default=120, type=int)
parser.add_argument('--learningRate', default=0.003, type=float)


args = parser.parse_args()

# Path
#=================================================================================================================
root_path = os.getcwd()
data_path = os.path.join(root_path, 'C_Data', 'static_networks')
dataloader_path = os.path.join(data_path, 'dataloaders')
result_path = os.path.join(root_path, 'D_Result', args.resultFolderName);  os.makedirs(result_path, exist_ok=True)


# Paramters
#=================================================================================================================
input_dim = 101
savePrefix = args.dataloaderType
common_cols = ['predW', 'groundTruthW', 'sInfo']

# dataloader
train_loader, val_loader, test_loader = load_dataloaders(dataloader_path, args.dataloaderName)
dl = load_oneLoader(dataloader_path, args.unseen_dlName)

# other params
cmap= 'coolwarm'; cm_cmap='PuBu'; 
mC='firebrick'; bC='steelblue'; ccgColor='k'
coolwarm_blue = '#3B4CC0'; coolwarm_red = '#B40426'
coolwarm_softblue = '#6BAED6'; coolwarm_softred = '#E07A7A'; correctGREEN= '#2E7D32'
coolwarm_blues = ['#7b9ff9', '#5977e3', '#3b4cc0']; coolwarm_reds = ['#ee8468', '#d65244', '#b40426']



# ConnCNN
#=================================================================================================================
best_val = math.inf; 
val_loss_runs = []; e, p, lr = args.trainingEpochs, args.trainingPatience, args.learningRate

for i in range(args.numRuns):
    # initialize ------------------------------------------------------------------------------------
    if savePrefix in ['Raw', 'batchScale', 'Scale']:
        model = WeightCNN(input_dim)
    elif savePrefix == 'Dual':
        model = WeightCNN_DualInput(input_dim)
    
    criterion = HuberLossWithWeight()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5,  betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    
    # training --------------------------------------------------------------------------------------
    _, epoch_loss_trace, val_loss_trace = trainWeightCNN_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=e, patience=p)
    val_loss  = float(np.min(val_loss_trace)); val_loss_runs.append(round(val_loss, 4))

    # testing ---------------------------------------------------------------------------------------
    test_result = evaluateWeightCNN_model(model, test_loader, criterion)
    test_loss = test_result[4]

    # picking best model ----------------------------------------------------------------------------
    if val_loss < best_val:
        print(f"-caught better VAl model, best_val_loss improved from {float(best_val):.4f} to {float(val_loss):.4f}  and test_loss = {float(test_loss):.4f} ")
        best_val = val_loss; best_val_model = model; best_val_result = test_result; best_valModel_testLoss = test_loss

# Best Model
#=================================================================================================================
# best val model -------------------------------------------------------------------------------------
avg_wegiht_loss, predW_list, gtW_list, label0_loss, label1_loss, sInfo_list = best_val_result
predW_list, gtW_list, sInfo_list = np.array(predW_list), np.array(gtW_list), np.array(sInfo_list)
m0 = (gtW_list == 0); mp = (gtW_list > 0); mn = (gtW_list < 0); m_nonzero = (gtW_list != 0)

lossFileName = f'bestVal_valLoss{float(best_val):.4f}_testLoss{float(best_valModel_testLoss):.4f}'

# plot --------------------------------------
plt.figure(figsize=(6, 6))

x = np.linspace(-1, 3, 100); plt.plot(x, x, linestyle="--", c=ccgColor, alpha=0.8)
plt.axvline(x=0, color='k', linestyle='-', linewidth=1)
plt.axhline(y=0, color='k', linestyle='-', linewidth=1)

plt.scatter(gtW_list[m0], predW_list[m0], c=ccgColor, alpha=0.4, s=120)
plt.scatter(gtW_list[mp], predW_list[mp], c=coolwarm_red, marker='x', alpha=0.6, s=40)
plt.scatter(gtW_list[mn], predW_list[mn], c=coolwarm_blue, marker='x', alpha=0.6, s=40)

plt.tight_layout(); plt.savefig(os.path.join(result_path, f'w{savePrefix}_{lossFileName}_trainingCM.png')); plt.close()

# save --------------------------------------
save_model(best_val_model, optimizer, scheduler, epoch_loss_trace, val_loss_trace, 
           avg_wegiht_loss, label0_loss, label1_loss, predW_list, gtW_list, 
           savePath=result_path, filename=f'w{savePrefix}_{lossFileName}.pth')


# Preds on simulated data
#=================================================================================================================
# on Networkloader 1 ----------------------------------------------------------------------------------
criterion = HuberLossWithWeight()
bestVal_pred_result = evaluateWeightCNN_model(best_val_model, dl, criterion)
avg_wegiht_loss, predW_list, gtW_list, label0_loss, label1_loss, sInfo_list = bestVal_pred_result
result_val = pd.DataFrame([predW_list, gtW_list, sInfo_list], index=[f'bestVal_{col}' for col in common_cols] ).T
print(f'best_val_model predict on networkLoader1.pkl got loss = {float(label1_loss):.4f}')

# bestMs stats on simulated data
#=================================================================================================================
criterion = HuberLossWithWeight()
bestVal_val_result = evaluateWeightCNN_model(best_val_model, val_loader, criterion)
bestVal_test_result = evaluateWeightCNN_model(best_val_model, test_loader, criterion)


