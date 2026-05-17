"""
# 05_make_one_dataloader.py
# Author: Xiaoqian Sun, 09/30/2025
# Fucntion: generate one dl with all samples from one network
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

import warnings
warnings.filterwarnings('ignore')

# arguments
#=================================================================================================================
parser = argparse.ArgumentParser("Static Network DLs ")
parser.add_argument('--dataFolderName', default='data', type=str)
parser.add_argument('--segFolderName', default='static_network', type=str)
parser.add_argument('--networkIdx', default='5', type=str)

args = parser.parse_args()

networkIdx = args.networkIdx

# path
#=================================================================================================================
root_path = str(PROJECT_ROOT)
data_path = os.path.join(root_path, args.dataFolderName)
segments_path = os.path.join(data_path, args.segFolderName)
ccgResult_path = os.path.join(segments_path, 'CCG')
dataloader_path = os.path.join(data_path, 'dataloaders'); os.makedirs(dataloader_path, exist_ok=True)


# load
#=================================================================================================================
# raw
CCH_df = pd.read_csv(os.path.join(ccgResult_path, 'CCH.csv'))

ccg_data = CCH_df[0:-1].T.to_numpy()
scaled_ccg_data = scale_ccg_baseline_batch(ccg_data, baseline_bins=10)

X = CCH_df.iloc[:-1].T.to_numpy().astype(np.float32)
scaler = StandardScaler(with_mean=True, with_std=True)
batchScaled_ccg_data = scaler.fit_transform(X) 


weights= CCH_df.iloc[-1:].to_numpy()[0]
presence = np.array([0 if l==0 else 1 for l in weights])
sample_info = np.array(CCH_df.columns.tolist())

one_dataloader(ccg_data, presence, weights, sample_info, batch_size=32, seed=42, 
               save_path=dataloader_path, filename=f'oneDL_{networkIdx}.pkl')
one_dataloader(scaled_ccg_data, presence, weights, sample_info, batch_size=32, seed=42, 
               save_path=dataloader_path, filename=f'oneDL_{networkIdx}_Scale.pkl')
one_dataloader(batchScaled_ccg_data, presence, weights, sample_info, batch_size=32, seed=42, 
               save_path=dataloader_path, filename=f'oneDL_{networkIdx}_batchScale.pkl')
one_dualchannel_dataloader(ccg_data, presence, weights, sample_info, 
                           batch_size=32, seed=42, scale_type="sample", baseline_bins=10,
                           save_path=dataloader_path, filename=f'oneDL_Dual_{networkIdx}.pkl')
one_dualchannel_dataloader(ccg_data, presence, weights, sample_info, 
                           batch_size=32, seed=42, scale_type="batch", baseline_bins=10,
                           save_path=dataloader_path, filename=f'oneDL_Dual{networkIdx}_batchScale.pkl')

print(f'Done one loaders for {args.segFolderName} saved in {dataloader_path} - oneDL_{networkIdx}.pkl....\n')


