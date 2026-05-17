"""
# 07_train_weight_cnn.py
# Author: Xiaoqian Sun, 10/03/2025
# Function: Train network on pooled (raw/scaled) network ccgs and predict on another unseen network - python scripts/07_train_weight_cnn.py --help
"""


# Import Packages
#=================================================================================================================
import os
import math
import argparse
import numpy as np
import pandas as pd
from Functions import *
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "PointNeuron_Simulation"))

import warnings
warnings.filterwarnings('ignore')


# arguments
#=================================================================================================================
parser = argparse.ArgumentParser("Training WeightCNN")

parser.add_argument('--dataloaderType', default='Raw', type=str, help='choose from Raw/Scale/Dual') 
parser.add_argument('--dataloaderName', default='TTV_2_3.pkl', type=str)
parser.add_argument('--resultFolderName', default='test1', type=str)

parser.add_argument('--inputDim', default=100, type=int)
parser.add_argument('--unseen_dlName', default='oneDL_1.pkl', type=str) 

parser.add_argument('--numRuns', default=30, type=int)
parser.add_argument('--trainingEpochs', default=120, type=int)
parser.add_argument('--trainingPatience', default=120, type=int)
parser.add_argument('--learningRate', default=0.003, type=float)

args = parser.parse_args()

# Path
#=================================================================================================================
root_path = str(PROJECT_ROOT)
dataloader_path = os.path.join(root_path, 'data', 'dataloaders')
result_path = os.path.join(root_path, 'results', args.resultFolderName);  os.makedirs(result_path, exist_ok=True)


# Paramters
#=================================================================================================================
input_dim = args.inputDim
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
e, p, lr = args.trainingEpochs, args.trainingPatience, args.learningRate

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
    val_loss  = float(np.min(val_loss_trace))

    # testing ---------------------------------------------------------------------------------------
    test_result = evaluateWeightCNN_model(model, test_loader, criterion)
    test_loss = test_result[4]

    # picking best model ----------------------------------------------------------------------------
    if val_loss < best_val:
        print(f"--caught better VAl model, best_val_loss improved from {float(best_val):.4f} to {float(val_loss):.4f}  and test_loss = {float(test_loss):.4f} ")
        best_val = val_loss
        best_val_model = model
        best_val_result = test_result 
        best_val_optimizer = optimizer
        best_val_scheduler = scheduler
        best_valModel_testLoss = test_loss
        best_val_val_loss_trace = val_loss_trace
        best_val_epoch_loss_trace = epoch_loss_trace


# best model 
# -------------------------------------------------------------------------------------
avg_weight_loss, predW_list, gtW_list, label0_loss, label1_loss, sInfo_list = best_val_result
predW_list, gtW_list, sInfo_list = np.array(predW_list), np.array(gtW_list), np.array(sInfo_list)
result_test = pd.DataFrame([predW_list, gtW_list, sInfo_list], index=common_cols).T
result_test.to_csv(os.path.join(result_path, 'w_heldOut_Test.csv'))

m0 = (gtW_list == 0); mp = (gtW_list > 0); mn = (gtW_list < 0); m_nonzero = (gtW_list != 0)
lossFileName = f'bestVal_valLoss{float(best_val):.4f}_testLoss{float(best_valModel_testLoss):.4f}'

# plot 
# -------------------------------------------------------------------------------------
plt.figure(figsize=(6, 6))

x = np.linspace(-1, 3, 100); plt.plot(x, x, linestyle="--", c=ccgColor, alpha=0.8)
plt.axvline(x=0, color='k', linestyle='-', linewidth=1)
plt.axhline(y=0, color='k', linestyle='-', linewidth=1)

plt.scatter(gtW_list[m0], predW_list[m0], c=ccgColor, alpha=0.4, s=120)
plt.scatter(gtW_list[mp], predW_list[mp], c=coolwarm_red, marker='x', alpha=0.6, s=40)
plt.scatter(gtW_list[mn], predW_list[mn], c=coolwarm_blue, marker='x', alpha=0.6, s=40)

plt.tight_layout(); plt.savefig(os.path.join(result_path, f'w{savePrefix}_{lossFileName}.png')); plt.close()

# save 
# -------------------------------------------------------------------------------------
save_model(best_val_model, best_val_optimizer, best_val_scheduler, 
           best_val_epoch_loss_trace, best_val_val_loss_trace, avg_weight_loss, 
           label0_loss, label1_loss, predW_list, gtW_list, savePath=result_path, filename=f'w{savePrefix}_{lossFileName}.pth')

# Preds on unseen
# -------------------------------------------------------------------------------------
criterion = HuberLossWithWeight()
bestVal_pred_result = evaluateWeightCNN_model(best_val_model, dl, criterion)
avg_weight_loss, predW_list, gtW_list, label0_loss, label1_loss, sInfo_list = bestVal_pred_result
result_unseen = pd.DataFrame([predW_list, gtW_list, sInfo_list], index=[f'bestVal_{col}' for col in common_cols] ).T
result_unseen.to_csv(os.path.join(result_path, 'w_unseen.csv'))

print(f'--best_val_model predict on {args.unseen_dlName} got avg_weight_loss = {float(avg_weight_loss):.4f}\n')



