"""
# 02_make_m21_dataloaders.py
# Author: Xiaoqian Sun, 10/01/2025
# Function: make many2one dataloaders
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


# argumetns
#=================================================================================================================
parser = argparse.ArgumentParser("Replicates Generation")
parser.add_argument('--dataFolderName', default='data', type=str)
parser.add_argument('--segFolderName', default='base', type=str)

parser.add_argument('--genTTVDL', action='store_true', help='include --genTTVDL in script, code will generate dataloader that contains TTV')
parser.add_argument('--genOneDL', action='store_true', help='include --genOneDL in script, code will generate dataloader that does not split TTV')

args = parser.parse_args()


# path
#=================================================================================================================
root_path = str(PROJECT_ROOT)
data_path = os.path.join(root_path, args.dataFolderName)
dataloader_path = os.path.join(data_path, 'dataloaders');  os.makedirs(dataloader_path, exist_ok=True)


# combine CCGs from multiple simulation, e.g., base_1, base_2, base_3...
#=================================================================================================================
ccg_list = []
for folder in os.listdir(data_path):
    if folder.startswith(args.segFolderName) and not folder.endswith('csv') :
        ccg = pd.read_csv(os.path.join(data_path, folder, 'CCG', 'CCH.csv')).iloc[:, 1:]
        ccg_columns = ccg.columns.tolist()
        ccg_new_columns = [folder+'_'+col for col in ccg_columns]
        ccg.columns = ccg_new_columns
        ccg_list.append(ccg)

CCH_df = pd.concat(ccg_list, axis=1)
CCH_df.to_csv(os.path.join(data_path, f'{args.segFolderName}_CCG.csv'))
print(f"--Done saving {os.path.join(data_path, f'{args.segFolderName}_CCG.csv')} with shape {ccg.shape} ")


# prepare
#=================================================================================================================
ccg_data = CCH_df[0:-1].T.to_numpy()
scaled_ccg_data = scale_ccg_baseline_batch(ccg_data, baseline_bins=10)

X = CCH_df.iloc[:-1].T.to_numpy().astype(np.float32)
scaler = StandardScaler(with_mean=True, with_std=True)
batchScaled_ccg_data = scaler.fit_transform(X) 

weights= CCH_df.iloc[-1:].to_numpy()[0]
presence = np.array([0 if l==0 else 1 for l in weights])
sample_info = np.array(CCH_df.columns.tolist())

if args.genOneDL: # generate one dataloader (for testing)
    one_dataloader(ccg_data, presence, weights, sample_info, batch_size=32, seed=42, 
                save_path=dataloader_path, filename=f'oneLoader_{args.segFolderName}.pkl')
    one_dataloader(scaled_ccg_data, presence, weights, sample_info, batch_size=32, seed=42, 
                save_path=dataloader_path, filename=f'oneLoader_{args.segFolderName}_scale.pkl')
    one_dataloader(batchScaled_ccg_data, presence, weights, sample_info, batch_size=32, seed=42, 
                save_path=dataloader_path, filename=f'oneLoader_{args.segFolderName}_batchScale.pkl')
    one_dualchannel_dataloader(ccg_data, presence, weights, sample_info, 
                            batch_size=32, seed=42, scale_type="sample", baseline_bins=10,
                            save_path=dataloader_path, filename=f'oneLoader_{args.segFolderName}_Dual.pkl')
    one_dualchannel_dataloader(ccg_data, presence, weights, sample_info, 
                            batch_size=32, seed=42, scale_type="batch", baseline_bins=10,
                            save_path=dataloader_path, filename=f'oneLoader_{args.segFolderName}_BatchDual.pkl')
    print(f'--Done one loaders for {args.segFolderName}\n')

elif args.genTTVDL: # generate TTV dataloader (for training)
    tl, vl, testl = prepare_dataloaders(ccg_data, presence, weights, sample_info, 
                                        batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_0_ratio=1.0, seed=42, 
                                        save_path=dataloader_path, filename=f'ttvLoader_{args.segFolderName}.pkl')
    tl, vl, testl = prepare_dataloaders(scaled_ccg_data, presence, weights, sample_info, 
                                        batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_0_ratio=1.0, seed=42, 
                                        save_path=dataloader_path, filename=f'ttvLoader_{args.segFolderName}_scale.pkl')
    tl, vl, testl = prepare_dataloaders(ccg_data, presence, weights, sample_info, 
                                        batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_0_ratio=1.0, seed=42, 
                                        ifBatchScale=True, save_path=dataloader_path, filename=f'ttvLoader_{args.segFolderName}_batchScale.pkl')
    tl, vl, testl = prepare_dualchannel_dataloaders(ccg_data, presence, weights, sample_info,
                                                    batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                                    num_0_ratio=1.0, seed=42, scale_type="sample", baseline_bins=10, 
                                                    save_path=dataloader_path, filename=f'ttvLoader_{args.segFolderName}_Dual.pkl')
    tl, vl, testl = prepare_dualchannel_dataloaders(ccg_data, presence, weights, sample_info, 
                                                    batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                                    num_0_ratio=1.0, seed=42, scale_type="batch", baseline_bins=10, 
                                                    save_path=dataloader_path, filename=f'ttvLoader_{args.segFolderName}_BatchDual.pkl')
    print(f'--Done TTV loaders for {args.segFolderName}\n')

else:
    print(f'--No dls generated\n')




