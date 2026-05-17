"""
# 04_make_network_dataloaders.py
# Author: Xiaoqian Sun, 10/01/2025
# Function: work on one network's CCGs or concatenate networks' CCGs
    - generate all kinds of train/test/val dataloaders
    - raw/scaled/dual-input
    - python scripts/04_make_network_dataloaders.py --help
"""



# Import Packages
#=================================================================================================================
import os
import argparse
import numpy as np
import pandas as pd
from Functions import *

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "PointNeuron_Simulation"))

from Functions import *

import warnings
warnings.filterwarnings('ignore')

# arguments
#=================================================================================================================
parser = argparse.ArgumentParser("Static Network DLs ")
parser.add_argument('--dataFolderName', default='data', type=str)
parser.add_argument('--segFolderNamePrefix', default='static_network', type=str)
parser.add_argument('--networkId_List', nargs='+', type=int, default=[1, 2, 3], help='List of network IDs')

args = parser.parse_args()

# path
#=================================================================================================================
root_path = str(PROJECT_ROOT)
data_path = os.path.join(root_path, args.dataFolderName)
dataloader_path = os.path.join(data_path, 'dataloaders'); os.makedirs(dataloader_path, exist_ok=True)


# Cconcatenate
#=================================================================================================================
targets = args.networkId_List; targetName = "_".join(map(str, targets))
all_ccgs = []; allScale_ccgs = []; all_weights = []; all_presence = []; all_info = []
for predictIdx in targets:
    try:
        ccgResult_path = os.path.join(data_path, f'{args.segFolderName}_{predictIdx}', 'CCG')

        CCH_df = pd.read_csv(os.path.join(ccgResult_path, 'CCH.csv')).iloc[:, 1:]
        
        ccg_data = CCH_df.iloc[:-1].T.to_numpy()  # (N_samples, N_bins)
        scaled_ccg_data = scale_ccg_baseline_batch(ccg_data, baseline_bins=10)

        weights = CCH_df.iloc[-1].to_numpy()     
        presence = np.array([0 if l == 0 else 1 for l in weights])
        sample_info = np.array([f"net{predictIdx}_{s}" for s in CCH_df.columns.tolist()])

        all_ccgs.append(ccg_data); allScale_ccgs.append(scaled_ccg_data); 
        all_weights.append(weights); all_presence.append(presence); all_info.append(sample_info)

    except Exception as e:
        print(predictIdx, '--', e)

# concat across samples
X = np.vstack(all_ccgs)             # (sum N_samples, N_bins)
X_Scaled = np.vstack(allScale_ccgs) # (sum N_samples, N_bins)
Y = np.concatenate(all_presence)    # (sum N_samples,)
W = np.concatenate(all_weights)     # (sum N_samples,)
S = np.concatenate(all_info)        # (sum N_samples,)
print(X.shape, W.shape, Y.shape, S.shape)



# raw -------------------------------------------------------
tl, vl, testl = prepare_dataloaders(X, Y, W, S, 
                                    #ccg_data, presence, weights, sample_info, 
                                    batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_0_ratio=1.0, seed=42, 
                                    save_path=dataloader_path, filename=f'TTV_{targetName}.pkl')
print(os.path.join(dataloader_path, f'TTV_{targetName}.pkl'), 'saved')

# batchScale ---------------------------------------------------
tl, vl, testl = prepare_dataloaders(X, Y, W, S, 
                                    #ccg_data, presence, weights, sample_info, 
                                    batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_0_ratio=1.0, seed=42, 
                                    ifBatchScale=True, save_path=dataloader_path, filename=f'TTV_{targetName}_batchScale.pkl')
print(os.path.join(dataloader_path, f'TTV_{targetName}_batchScale.pkl'), 'saved')


# scaled ---------------------------------------------------
tl, vl, testl = prepare_dataloaders(X_Scaled, Y, W, S, 
                                    #ccg_data, presence, weights, sample_info, 
                                    batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_0_ratio=1.0, seed=42, 
                                    save_path=dataloader_path, filename=f'TTV_{targetName}_Scale.pkl')
print(os.path.join(dataloader_path, f'TTV_{targetName}_Scale.pkl'), 'saved')


# raw+scale dual -------------------------------------------
tl, vl, testl = prepare_dualchannel_dataloaders(X, Y, W, S, 
                                                #raw_ccgs, connectivity, weights, sample_info
                                                batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                                num_0_ratio=1.0, seed=42, scale_type="sample", baseline_bins=10, 
                                                save_path=dataloader_path, filename=f'TTV_Dual_{targetName}.pkl')
print(os.path.join(dataloader_path, f'TTV_Dual_{targetName}_Scale.pkl'), 'saved')

# raw+batchScale dual -------------------------------------------
tl, vl, testl = prepare_dualchannel_dataloaders(X, Y, W, S, 
                                                #raw_ccgs, connectivity, weights, sample_info
                                                batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                                num_0_ratio=1.0, seed=42, scale_type="batch", baseline_bins=10, 
                                                save_path=dataloader_path, filename=f'TTV_Dual_{targetName}_batchScale.pkl')
print(os.path.join(dataloader_path, f'TTV_Dual_{targetName}_BatchScale.pkl'), 'saved')


print(f'Done.\n')


