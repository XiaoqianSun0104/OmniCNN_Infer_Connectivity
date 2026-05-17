"""
# 06_train_connectivity_cnn.py
# Author: Xiaoqian Sun, 10/03/2025
# Function: Train network on pooled (raw/scaled) network ccgs and predict on another unseen network - python scripts/06_train_connectivity_cnn.py --help
"""

# Import Packages
#=================================================================================================================
import os
import argparse
import numpy as np
import pandas as pd
from Functions import *
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
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
parser = argparse.ArgumentParser("Training ConnCNN")

parser.add_argument('--dataFolderName', default='data', type=str)

parser.add_argument('--dataloaderType', default='Raw', type=str, help='choose from Raw/Scale/Dual') 
parser.add_argument('--dataloaderName', default='TTV_2_3.pkl', type=str)
parser.add_argument('--resultFolderName', default='test1', type=str)

parser.add_argument('--inputDim', default=101, type=int)
parser.add_argument('--unseen_dlName', default='oneDL_1.pkl', type=str) 

parser.add_argument('--numRuns', default=30, type=int)
parser.add_argument('--trainingEpochs', default=120, type=int)
parser.add_argument('--trainingPatience', default=120, type=int)
parser.add_argument('--learningRate', default=0.003, type=float)


args = parser.parse_args()

# Path
#=================================================================================================================
root_path = str(PROJECT_ROOT)
data_path = os.path.join(root_path, args.dataFolderName)
dataloader_path = os.path.join(data_path, 'dataloaders')
result_path = os.path.join(root_path, 'results', args.resultFolderName);  os.makedirs(result_path, exist_ok=True)


# Paramters
#=================================================================================================================
input_dim = args.inputDim
savePrefix = args.dataloaderType
common_cols = ['pred', 'confi', 'label', 'weights', 'sInfo']

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
best_val = -1.0
e, p, lr = args.trainingEpochs, args.trainingPatience, args.learningRate

for i in range(args.numRuns):
    set_seed(i)

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
    val_acc  = float(np.max(val_acc_trace))

    # testing ---------------------------------------------------------------------------------------
    test_result = evaluateConnCNN_wConfi_model(model, test_loader, criterion)
    test_acc = test_result[1]

    # picking best model ----------------------------------------------------------------------------
    if val_acc > best_val:
        print(f"--caught better VAl model, best_val_accu improved from {float(best_val):.4f} to {float(val_acc):.4f}  and test_accu = {float(test_acc):.4f} ")
        best_val = val_acc
        best_val_model = model
        best_val_result = test_result
        best_val_optimizer = optimizer
        best_val_scheduler = scheduler
        best_valModel_testAccu = test_acc
        best_val_val_loss_trace = val_loss_trace
        best_val_epoch_loss_trace = epoch_loss_trace


# best model 
# -------------------------------------------------------------------------------------
avg_connect_loss, accu, pred_list, confi_list, label_list, w_list, sInfo_list = best_val_result
result_test = pd.DataFrame([pred_list, confi_list, label_list, w_list, sInfo_list], index=['pred', 'confi', 'label', 'weights', 'sInfo'] ).T
result_test.to_csv(os.path.join(result_path, 'c_heldOut_Test.csv'))

accuFileName = f'bestVal_valAccu{float(best_val):.4f}_testAccu{float(accu):.4f}'

# plot 
# -------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].plot(best_val_epoch_loss_trace, c=mC, label='training loss'); ax[0].plot(best_val_val_loss_trace, c=bC, label='validation loss')
ax[0].legend(); ax[0].set_title('Training/Val Epoch Loss', fontweight='bold')

CM = confusion_matrix(label_list, pred_list)
sns.heatmap(CM, annot=True, fmt="d", cmap='PuBu', ax=ax[1],
            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
ax[1].set_title('Accu='+str(round(accu, 4)), fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(result_path, f'c{savePrefix}_{accuFileName}_trainingCM.png')); plt.close()

# save 
# -------------------------------------------------------------------------------------
save_conn_model(best_val_model, best_val_optimizer, best_val_scheduler, 
                best_val_epoch_loss_trace, best_val_val_loss_trace, avg_connect_loss, accu, pred_list, 
                label_list, w_list, sInfo_list, savePath=result_path, filename=f'c{savePrefix}_{accuFileName}.pth')

# Preds on unseen
# -------------------------------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
bestVal_pred_result = evaluateConnCNN_wConfi_model(best_val_model, dl, criterion)
avg_connect_loss, accu, pred_list, confi_list, label_list, w_list, sInfo_list = bestVal_pred_result
result_unseen = pd.DataFrame([pred_list, confi_list, label_list, w_list, sInfo_list], index=['pred', 'confi', 'label', 'weights', 'sInfo'] ).T
result_unseen.to_csv(os.path.join(result_path, 'c_unseen.csv'))

print(f'--best_val_model predict on {args.unseen_dlName} got accu = {float(accu):.4f}\n')

