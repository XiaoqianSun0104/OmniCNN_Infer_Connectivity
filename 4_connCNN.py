# 5_connCNN.py
# Author: Xiaoqian Sun, 10/03/2025
# Fucntion: Train network on pooled (raw/scaled) network ccgs and predict on another unseen network. 
#           Try to see if MCC can be improved if trained on networks cross domains


# Import Packages
#=================================================================================================================
import os
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
from utils import *; from visualization import *

import Functions
importlib.reload(Functions)
from Functions import *

import warnings
warnings.filterwarnings('ignore')


# argumetns
#=================================================================================================================
parser = argparse.ArgumentParser("Traning ConnCNN")

parser.add_argument('--dataloaderType', default='Raw', type=str)          # Raw/Scale/Dual
parser.add_argument('--dataloaderName', default='networkLoader_2_3_4_5_6_7_TTV.pkl', type=str)
parser.add_argument('--resultFolderName', default='networks_2_3_4_5_6_7', type=str)

parser.add_argument('--unseen_dlName', default='networkLoader1.pkl', type=str)  #networkLoader_scale.pkl
parser.add_argument('--eANN_dlName', default='eANNLoader.pkl', type=str)  #eANNLoader_scale.pkl

parser.add_argument('--numRuns', default=30, type=int)
parser.add_argument('--trainingEpochs', default=120, type=int)
parser.add_argument('--trainingPatience', default=120, type=int)
parser.add_argument('--learningRate', default=0.003, type=float)


args = parser.parse_args()

# Path
#=================================================================================================================
root_path = r'/CCAS/home/xiaoqian10/inferConnectivity_Simulation'
data_path = os.path.join(root_path, 'C_Data', 'static_networks')
dataloader_path = os.path.join(data_path, 'dataloaders')
result_path = os.path.join(root_path, 'D_Result', args.resultFolderName);  os.makedirs(result_path, exist_ok=True)


# Paramters
#=================================================================================================================
input_dim = 101
savePrefix = args.dataloaderType
common_cols = ['pred', 'confi', 'label', 'weights', 'sInfo']

# dataloader
train_loader, val_loader, test_loader = load_dataloaders(dataloader_path, args.dataloaderName)
dl = load_oneLoader(dataloader_path, args.unseen_dlName)
eANN_dl = load_oneLoader(dataloader_path, args.eANN_dlName)

# other params
cmap= 'coolwarm'; cm_cmap='PuBu'; 
mC='firebrick'; bC='steelblue'; ccgColor='k'
coolwarm_blue = '#3B4CC0'; coolwarm_red = '#B40426'
coolwarm_softblue = '#6BAED6'; coolwarm_softred = '#E07A7A'; correctGREEN= '#2E7D32'
coolwarm_blues = ['#7b9ff9', '#5977e3', '#3b4cc0']; coolwarm_reds = ['#ee8468', '#d65244', '#b40426']



# ConnCNN
#=================================================================================================================
best_val = -1.0
val_acc_runs = []; e, p, lr = args.trainingEpochs, args.trainingPatience, args.learningRate

for i in range(args.numRuns):
    # initialize ------------------------------------------------------------------------------------
    if savePrefix in ('Raw', 'Scale'):
        model = ConnCNN(input_dim)
    elif savePrefix == 'Dual':
        model = ConnCNN_DualInput(input_dim)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))  # lr=0.005
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
    
    
    # training --------------------------------------------------------------------------------------
    _, epoch_loss_trace, val_loss_trace, val_acc_trace = trainConnCNN_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=e, patience=p )
    val_acc  = float(np.max(val_acc_trace)); val_acc_runs.append(round(val_acc, 4))

    # testing ---------------------------------------------------------------------------------------
    test_result = evaluateConnCNN_wConfi_model(model, test_loader, criterion)
    test_acc = test_result[1]

    # picking best model ----------------------------------------------------------------------------
    if val_acc > best_val:
        print(f"-caught better VAl model, best_val_accu improved from {float(best_val):.4f} to {float(val_acc):.4f}  and test_accu = {float(test_acc):.4f} ")
        best_val = val_acc; best_val_model = model; best_val_result = test_result; best_valModel_testAccu = test_acc


# Best Model
#=================================================================================================================
# best val model -------------------------------------------------------------------------------------
avg_connect_loss, accu, pred_list, confi_list, label_list, w_list, sInfo_list = best_val_result
accuFileName = f'bestVal_valAccu{float(best_val):.4f}_testAccu{float(accu):.4f}'

# plot --------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].plot(epoch_loss_trace, c=mC, label='epoch_loss'); ax[0].plot(val_loss_trace, c=bC, label='val_loss')
ax[0].legend(); ax[0].set_title('Training Epoch Loss', fontweight='bold')

CM = confusion_matrix(label_list, pred_list)
sns.heatmap(CM, annot=True, fmt="d", cmap='PuBu', ax=ax[1],
            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
ax[1].set_title('Accu='+str(round(accu, 4)), fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(result_path, f'c{savePrefix}_{accuFileName}_trainingCM.png')); plt.close()

# save --------------------------------------
save_conn_model(best_val_model, optimizer, scheduler, epoch_loss_trace, val_loss_trace, avg_connect_loss, accu, pred_list, 
                label_list, w_list, sInfo_list, savePath=result_path, filename=f'c{savePrefix}_{accuFileName}.pth')

# Preds on simulated/eANN-real data
#=================================================================================================================
# on Networkloader 1 ----------------------------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
bestVal_pred_result = evaluateConnCNN_wConfi_model(best_val_model, dl, criterion)
avg_connect_loss, accu, pred_list, confi_list, label_list, w_list, sInfo_list = bestVal_pred_result
result_val = pd.DataFrame([pred_list, confi_list, label_list, w_list, sInfo_list], index=[f'bestVal_{col}' for col in common_cols] ).T
print(f'best_val_model predict on networkLoader1.pkl got accu = {float(accu):.4f}')

# on eANN dataloader ----------------------------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
bestVal_eANN_result = evaluateConnCNN_wConfi_model(best_val_model, eANN_dl, criterion)
avg_connect_loss, accu, pred_list, confi_list, label_list, w_list, sInfo_list = bestVal_eANN_result
result_val = pd.DataFrame([pred_list, confi_list, label_list, w_list, sInfo_list], index=[f'bestVal_{col}' for col in common_cols] ).T
print(f'best_val_model predict on eANN_dl.pkl got accu = {float(accu):.4f}')


