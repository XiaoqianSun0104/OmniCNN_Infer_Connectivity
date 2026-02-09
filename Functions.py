# Functions.py
#
# Author: Xiaoqian Sun, 03/2025
#
# Functions used in ML Models


# IMPORT
import os
import copy
import random
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.stats import spearmanr
from scipy.special import rel_entr
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_curve, confusion_matrix, classification_report



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import warnings
warnings.filterwarnings('ignore')



# FUNCTIONS

# dataloaders ==============================================================================================
# Note: SimulationDataset()&prepare_dataloaders() have been modified to include sample info in the dataloader
class SimulationDataset(Dataset):
    def __init__(self, ccgs, connectivity, weights, sample_info):
        
        '''
        Data:
            - ccgs: NumPy array of shape (N, T) -
            - connectivity: NumPy array of shape (N,) - Binary 0/1
            - weights: NumPy array of shape (N,) - Synaptic weights (floats)
            - sample_info: sample metadata-neuron name (e,g,m 'SE0_SE9', 'SI1_SE42' )
        '''
        self.ccgs = ccgs
        self.connectivity = connectivity.unsqueeze(1)
        self.weights = weights.unsqueeze(1)
        self.sample_info = sample_info  # list of strings
    
    def __len__(self):
        return len(self.ccgs)
    
    def __getitem__(self, idx):
        return self.ccgs[idx], self.connectivity[idx], self.weights[idx], self.sample_info[idx]
    
def prepare_dataloaders(ccgs, connectivity, weights, sample_info, batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, num_0_ratio=1.0, seed=42, ifBatchScale=False, save_path=None, filename=None):
    '''
    This function does following things:
        - since connectivity data is highly imbalanced, we use num_0_ratio to control the ration betweel num_0 and num_1
        - make sure proportion of labels (0/1) is preserved acorss train/test/validation
            - e.g., 100 samples with label 1; 70-in train, 15-in test, 15-in validation
            - this is called `stratification`
    
    '''

    assert train_ratio + val_ratio + test_ratio == 1.0, "Splits must sum to 1.0"
    
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
    # 2 catogeries
    idx_1 = np.where(connectivity == 1)[0]
    idx_0 = np.where(connectivity == 0)[0]
    
    # subsample 0s
    num_1 = len(idx_1)
    num_0 = min(len(idx_0), int(num_1 * num_0_ratio))
    idx_0 = np.random.choice(idx_0, num_0, replace=False) 
    
    # merge & shuffle
    selected_idx = np.concatenate([idx_1, idx_0])
    np.random.shuffle(selected_idx)
    
    # subset
    ccgs_selected = torch.tensor(ccgs[selected_idx], dtype=torch.float32)
    connectivity_selected = torch.tensor(connectivity[selected_idx], dtype=torch.float32)
    weights_selected = torch.tensor(weights[selected_idx], dtype=torch.float32)
    sample_info_selected = sample_info[selected_idx]
    
    # train/test/validation & stratification
    stratify_labels = connectivity_selected.numpy()
    train_idx, temp_idx = train_test_split(np.arange(len(stratify_labels)), test_size=(1-train_ratio), stratify=stratify_labels, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_ratio/(test_ratio+val_ratio), stratify=stratify_labels[temp_idx], random_state=seed)

    # dataset-level per-bin scaling for weight inference
    if ifBatchScale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        ccgs_selected_np = np.asarray(ccgs[selected_idx], dtype=np.float32) 
        scaler.fit(ccgs_selected_np[train_idx])    # fit on TRAIN ONLY (no leakage)
        ccgs_selected_np[train_idx] = scaler.transform(ccgs_selected_np[train_idx])
        ccgs_selected_np[val_idx]   = scaler.transform(ccgs_selected_np[val_idx])
        ccgs_selected_np[test_idx]  = scaler.transform(ccgs_selected_np[test_idx])
        ccgs_selected = torch.tensor(ccgs_selected_np, dtype=torch.float32)

    # datasets
    train_set = SimulationDataset(ccgs_selected[train_idx], connectivity_selected[train_idx], weights_selected[train_idx],sample_info_selected[train_idx])
    val_set = SimulationDataset(ccgs_selected[val_idx], connectivity_selected[val_idx], weights_selected[val_idx], sample_info_selected[val_idx])
    test_set = SimulationDataset(ccgs_selected[test_idx], connectivity_selected[test_idx], weights_selected[test_idx], sample_info_selected[test_idx])

    # dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, filename), "wb") as f:
            pickle.dump({"train": train_loader, "val": val_loader, "test": test_loader}, f)
    
    print(f"Num Samples: Train {len(train_set)}, Val {len(val_set)}, Test {len(test_set)}")
    return train_loader, val_loader, test_loader
    
def one_dataloader(ccgs, connectivity, weights, sample_info, batch_size=32, seed=42, save_path=None, filename=None):
    '''
    only 1 dataloader which contains all data
    '''

    
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # datasets
    data_set = SimulationDataset(torch.tensor(ccgs, dtype=torch.float32),
                                 torch.tensor(connectivity, dtype=torch.float32),
                                 torch.tensor(weights, dtype=torch.float32), sample_info)
    
    # dataloader
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)

    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, filename), "wb") as f:
            pickle.dump({"data_loader": data_loader}, f)    





class DualSimulationDataset(Dataset):
    def __init__(self, ccgs_raw, ccgs_scaled, connectivity, weights, sample_info):
        '''
        Inputs:
            - ccgs_raw: (N, T)
            - ccgs_scaled: (N, T)
            - connectivity: (N,) - binary
            - weights: (N,) - synaptic weights
            - sample_info: list of N strings
        '''
        self.ccgs_raw = ccgs_raw.unsqueeze(1)      # [N, 1, T]
        self.ccgs_scaled = ccgs_scaled.unsqueeze(1)  # [N, 1, T]
        self.connectivity = connectivity.unsqueeze(1)
        self.weights = weights.unsqueeze(1)
        self.sample_info = sample_info

    def __len__(self):
        return len(self.ccgs_raw)

    def __getitem__(self, idx):
        ccg_dual = torch.cat([self.ccgs_raw[idx], self.ccgs_scaled[idx]], dim=0)  # [2, T]
        return ccg_dual, self.connectivity[idx], self.weights[idx], self.sample_info[idx]

def prepare_dualchannel_dataloaders(raw_ccgs, connectivity, weights, sample_info, 
                                    batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                                    num_0_ratio=1.0, seed=42, scale_type="batch", baseline_bins=10, save_path=None, filename=None):
    '''
    scale_type: "batch"  = scale all samples using scaler fit on training set (no leakage)
                "sample" = scale each sample individually
                 None    = skip scaling
    '''

    assert train_ratio + val_ratio + test_ratio == 1.0, "Splits must sum to 1.0"

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2 catogeries
    idx_1 = np.where(connectivity == 1)[0]
    idx_0 = np.where(connectivity == 0)[0]

    # subsample 0s
    num_1 = len(idx_1)
    num_0 = min(len(idx_0), int(num_1 * num_0_ratio))
    idx_0 = np.random.choice(idx_0, num_0, replace=False)

    # merge & shuffle
    selected_idx = np.concatenate([idx_1, idx_0])
    np.random.shuffle(selected_idx)

    # subset
    raw_ccgs = np.asarray(raw_ccgs[selected_idx], dtype=np.float32)
    connectivity = torch.tensor(connectivity[selected_idx], dtype=torch.float32)
    weights = torch.tensor(weights[selected_idx], dtype=torch.float32)
    sample_info = sample_info[selected_idx]

    # train/test/validation & stratification
    stratify_labels = connectivity.numpy()
    train_idx, temp_idx = train_test_split(np.arange(len(stratify_labels)), test_size=(1-train_ratio), stratify=stratify_labels, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_ratio/(val_ratio+test_ratio), stratify=stratify_labels[temp_idx], random_state=seed)

    # scaling starts here
    scaled_ccgs = np.copy(raw_ccgs)
    if scale_type == "batch":
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(raw_ccgs[train_idx])
        scaled_ccgs[train_idx] = scaler.transform(raw_ccgs[train_idx])
        scaled_ccgs[val_idx]   = scaler.transform(raw_ccgs[val_idx])
        scaled_ccgs[test_idx]  = scaler.transform(raw_ccgs[test_idx])
    elif scale_type == "sample":
        scaled_ccgs = scale_ccg_baseline_batch(raw_ccgs, baseline_bins=baseline_bins)

    # Convert to torch tensors
    raw_ccgs = torch.tensor(raw_ccgs, dtype=torch.float32)
    scaled_ccgs = torch.tensor(scaled_ccgs, dtype=torch.float32)

    train_set = DualSimulationDataset(raw_ccgs[train_idx], scaled_ccgs[train_idx], connectivity[train_idx], weights[train_idx], sample_info[train_idx])
    val_set   = DualSimulationDataset(raw_ccgs[val_idx],   scaled_ccgs[val_idx],   connectivity[val_idx],   weights[val_idx],   sample_info[val_idx])
    test_set  = DualSimulationDataset(raw_ccgs[test_idx],  scaled_ccgs[test_idx],  connectivity[test_idx],  weights[test_idx],  sample_info[test_idx])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, filename), "wb") as f:
            pickle.dump({"train": train_loader, "val": val_loader, "test": test_loader}, f)

    print(f"Num Samples: Train {len(train_set)}, Val {len(val_set)}, Test {len(test_set)}")
    return train_loader, val_loader, test_loader

def one_dualchannel_dataloader(raw_ccgs, connectivity, weights, sample_info,
                               batch_size=32, seed=42, scale_type="batch", baseline_bins=10,save_path=None, filename=None):
    """
    Creates a single DataLoader for the entire dataset using both raw and scaled CCGs.
    
    scale_type: 
        - "batch"  = scale all samples using scaler fit on full set
        - "sample" = scale each sample individually using baseline bins
        - None     = skip scaling (raw and scaled inputs are identical)
    """

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Ensure raw_ccgs is a float32 NumPy array
    raw_ccgs = np.asarray(raw_ccgs, dtype=np.float32)
    
    # Copy and scale
    scaled_ccgs = np.copy(raw_ccgs)

    if scale_type == "batch":
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(raw_ccgs)  # Fit on full data since no split
        scaled_ccgs = scaler.transform(raw_ccgs)
    elif scale_type == "sample":
        scaled_ccgs = scale_ccg_baseline_batch(raw_ccgs, baseline_bins=baseline_bins)
    elif scale_type is None:
        scaled_ccgs = np.copy(raw_ccgs)
    else:
        raise ValueError(f"Invalid scale_type: {scale_type}")

    # Convert to torch tensors
    raw_ccgs = torch.tensor(raw_ccgs, dtype=torch.float32)
    scaled_ccgs = torch.tensor(scaled_ccgs, dtype=torch.float32)
    connectivity = torch.tensor(connectivity, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)

    # Create Dataset and DataLoader
    data_set = DualSimulationDataset(raw_ccgs, scaled_ccgs, connectivity, weights, sample_info)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)

    # Optional save
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, filename), "wb") as f:
            pickle.dump({"data_loader": data_loader}, f)





def load_dataloaders(load_path, filename):
    '''
    load back preprocessed train/test/validation dataloaders
    '''
    with open(os.path.join(load_path, filename), "rb") as f:
        dataloaders = pickle.load(f)
    
    return dataloaders["train"], dataloaders["val"], dataloaders["test"]

def load_oneLoader (load_path, filename):
    with open(os.path.join(load_path, filename), "rb") as f:
        dataloaders = pickle.load(f)
    
    return dataloaders["data_loader"]

def sampleDistribution(dataloader):
    
    ''' check how many samples are non-connected, now many samples are exc/inh out of connected pairs '''

    css = []; wss = []
    for ccgs, cs, ws, _ in dataloader:
        css.append(cs); wss.append(ws)
    
    css = torch.cat(css, dim=0); wss = torch.cat(wss, dim=0)
    
    return {'noConn':(css==0).sum().item(), 'Conn': (css!=0).sum().item(),
            'excConn':(wss>0).sum().item(), 'excConn>1':(wss>1).sum().item(), 'excConn>2':(wss>2).sum().item(), 
            'inhConn':(wss<0).sum().item(), 'inhConn<-0.5':(wss<-0.5).sum().item(), 'inhConn<-1':(wss<-1).sum().item(),}

def prepare_wbal_dataloaders(ccgs, connectivity, weights, sample_info,
                        batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                        num_0_ratio=1.0, seed=42, save_path=None, filename=None):
    '''
    Prepares stratified dataloaders for inhibitory, excitatory, and no-connection synaptic weights.

    Ensures that train/val/test splits preserve the class distribution across:
        - inhibitory (<0)
        - no connection (==0)
        - excitatory (>0)
    '''
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Splits must sum to 1.0"

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -----------------------------
    # Step 1: Define weight labels
    # -----------------------------
    weight_classes = np.zeros_like(weights)
    weight_classes[weights > 0] = 1   # Excitatory
    weight_classes[weights < 0] = -1  # Inhibitory

    # -----------------------------
    # Step 2: Sample 0-weight examples
    # -----------------------------
    idx_exc = np.where(weight_classes == 1)[0]
    idx_inh = np.where(weight_classes == -1)[0]
    idx_none = np.where(weight_classes == 0)[0]

    num_connected = len(idx_exc) + len(idx_inh)
    num_0 = min(len(idx_none), int(num_connected * num_0_ratio))
    idx_none_sampled = np.random.choice(idx_none, num_0, replace=False)

    # Merge all selected indices
    selected_idx = np.concatenate([idx_exc, idx_inh, idx_none_sampled])
    np.random.shuffle(selected_idx)

    # -----------------------------
    # Step 3: Prepare data arrays
    # -----------------------------
    ccgs_selected = torch.tensor(ccgs[selected_idx], dtype=torch.float32)
    connectivity_selected = torch.tensor(connectivity[selected_idx], dtype=torch.float32)
    weights_selected = torch.tensor(weights[selected_idx], dtype=torch.float32)
    sample_info_selected = sample_info[selected_idx]

    # Shift weight class labels to 0/1/2 for sklearn
    stratify_labels = weight_classes[selected_idx] + 1  # -1 → 0, 0 → 1, 1 → 2

    # -----------------------------
    # Step 4: Stratified splits
    # -----------------------------
    train_val_ratio = train_ratio + val_ratio
    test_size = 1 - train_val_ratio

    train_val_idx, test_idx = train_test_split(
        np.arange(len(selected_idx)),
        test_size=test_size,
        stratify=stratify_labels,
        random_state=seed
    )

    # Stratify again on train/val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adjusted,
        stratify=stratify_labels[train_val_idx],
        random_state=seed
    )

    # -----------------------------
    # Step 5: Build Datasets
    # -----------------------------
    train_set = SimulationDataset(ccgs_selected[train_idx],
                                  connectivity_selected[train_idx],
                                  weights_selected[train_idx],
                                  sample_info_selected[train_idx])
    val_set = SimulationDataset(ccgs_selected[val_idx],
                                connectivity_selected[val_idx],
                                weights_selected[val_idx],
                                sample_info_selected[val_idx])
    test_set = SimulationDataset(ccgs_selected[test_idx],
                                 connectivity_selected[test_idx],
                                 weights_selected[test_idx],
                                 sample_info_selected[test_idx])

    # -----------------------------
    # Step 6: Dataloaders
    # -----------------------------
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # Step 7: Save if needed
    # -----------------------------
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, filename), "wb") as f:
            pickle.dump({"train": train_loader, "val": val_loader, "test": test_loader}, f)

    print(f"Num Samples: Train {len(train_set)}, Val {len(val_set)}, Test {len(test_set)}")

    return train_loader, val_loader, test_loader





# save/load model ==========================================================================================
def save_model(model, optimizer, scheduler, epoch_loss_trace, val_loss_trace, 
               avg_wegiht_loss, z0_loss, zn0_loss, predW_list, gtW_list, savePath, filename):
    
    '''
    save everything about the model. can be used to save
        - MLP_weight
        - 1D CNN_weight
    
    '''
    
    save_dict = {
        'model_state_dict': model.state_dict(),          # model parameters
        'optimizer_state_dict': optimizer.state_dict(),  # optimizer state
        'scheduler_state_dict': scheduler.state_dict(),  # learning rate scheduler
        
        # training
        'epoch_loss_trace': epoch_loss_trace,    # loss per epoch
        'val_loss_trace': val_loss_trace,        # loss per batch
        
        # testing
        'avg_wegiht_loss': avg_wegiht_loss,     # testloader, avg loss
        'lable0_loss': z0_loss,                 # avg loss for no-connection ccg
        'lable1_loss': zn0_loss,                # ~ for connected ccg
        'predW_list': predW_list,               # prediction
        'gtW_list': gtW_list,                   # ground truth
    }
    
    torch.save(save_dict, os.path.join(savePath, filename)); print('Done Saving the Model -', filename)
    
def load_model(emptyModel, emptyOptimizer, emptyScheduler, savePath, filename):
    
    '''
    load model back
    '''
    
    checkpoint = torch.load(os.path.join(savePath, filename), weights_only=False)
    
    emptyModel.load_state_dict(checkpoint['model_state_dict']); emptyModel.eval() 
    
    emptyOptimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    emptyScheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    val_loss_trace = checkpoint['val_loss_trace']
    epoch_loss_trace = checkpoint['epoch_loss_trace']
    

    avg_wegiht_loss = checkpoint['avg_wegiht_loss']
    z0_loss = checkpoint['lable0_loss']
    zn0_loss = checkpoint['lable1_loss']
    predW_list = checkpoint['predW_list']
    gtW_list = checkpoint['gtW_list']
    
    
    print('Model loaded successfully from', filename)
    
    return emptyModel, (emptyOptimizer, emptyScheduler, epoch_loss_trace, val_loss_trace, 
                   avg_wegiht_loss, z0_loss, zn0_loss, predW_list, gtW_list)

def save_conn_model(model, optimizer, scheduler, epoch_loss_trace, val_loss_trace, 
               avg_test_loss, accu, pred_list, label_list, w_list, sInfo_list, savePath, filename):
    
    '''
    save everything about the model. can be used to save
        - MLP_weight
        - 1D CNN_weight
    
    '''
    
    save_dict = {
        'model_state_dict': model.state_dict(),          # model parameters
        'optimizer_state_dict': optimizer.state_dict(),  # optimizer state
        'scheduler_state_dict': scheduler.state_dict(),  # learning rate scheduler
        
        # training
        'epoch_loss_trace': epoch_loss_trace,    # loss per epoch
        'val_loss_trace': val_loss_trace,        # loss per batch
        
        # testing
        'avg_test_loss': avg_test_loss,     # testloader, avg loss
        'accu': accu,                 # avg loss for no-connection ccg
        'pred_list': pred_list,                # ~ for connected ccg
        'label_list': label_list,               # prediction
        'w_list': w_list,
        'sInfo_list': sInfo_list,                   # ground truth
    }
    
    torch.save(save_dict, os.path.join(savePath, filename)); print('Done Saving the Model -',filename)
    
def load_conn_model(emptyModel, emptyOptimizer, emptyScheduler, savePath, filename):
    
    '''
    load model back
    '''
    
    checkpoint = torch.load(os.path.join(savePath, filename), weights_only=False)
    
    emptyModel.load_state_dict(checkpoint['model_state_dict']); emptyModel.eval() 
    
    emptyOptimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    emptyScheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    val_loss_trace = checkpoint['val_loss_trace']
    epoch_loss_trace = checkpoint['epoch_loss_trace']
    
    avg_test_loss = checkpoint['avg_test_loss']
    accu = checkpoint['accu']
    pred_list = checkpoint['pred_list']
    label_list = checkpoint['label_list']
    w_list = checkpoint['w_list']
    sInfo_list = checkpoint['sInfo_list']
    
    
    print('Model loaded successfully from', filename)
    
    return emptyModel, (emptyOptimizer, emptyScheduler, epoch_loss_trace, val_loss_trace, 
                   avg_test_loss, accu, pred_list, label_list, w_list, sInfo_list)





# CNN Define =============================================================================================
class ConnCNN(nn.Module):  # original
    def __init__(self, input_dim):
        super(ConnCNN, self).__init__()
        
        conv1_layer=4; kernel1_size=70
        conv2_layer=2; kernel2_size=35

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_layer, kernel_size=kernel1_size, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(conv1_layer)

        self.conv2 = nn.Conv1d(in_channels=conv1_layer, out_channels=conv2_layer, kernel_size=kernel2_size, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(conv2_layer)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)  # reduces sequence to a single vector
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # helps generalize better

        self.fc = nn.Linear(conv2_layer, 1)
        

    def forward(self, x):
        
        if x.dim() == 2: 
            x = x.unsqueeze(1)
        
        x = self.bn1(self.conv1(x))
        x = self.tanh(x)
        x = self.bn2(self.conv2(x))
        x = self.tanh(x)              # torch.Size([32, 2, 6])
        
        x = self.global_avg_pool(x)   # torch.Size([32, 2, 1])
        x = x.squeeze(-1)             # torch.Size([32, 2])
        x = self.dropout(x)

        # x = self.fc(x)                # torch.Size([32, 1])
        # conn = self.sigmoid(x)
        conn = self.fc(x)  

        return conn

def trainConnCNN_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, patience=3):
    model.train()

    best_val_loss = float('inf'); counter=0; loss_trace = []; epoch_loss_trace = []; val_loss_trace = []; val_acc_trace = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for ccgs, labels, weights, _ in train_loader:
            optimizer.zero_grad()

            connectivity_out = model(ccgs)
            loss = criterion(connectivity_out, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss = loss.item() * ccgs.size(0)
            running_loss += batch_loss
            loss_trace.append(batch_loss)


        # average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_loss_trace.append(epoch_loss)

        # dynamically adjust learning rate
        val_loss,val_acc,_,_,_,_ = evaluateConnCNN_model(model, val_loader, criterion)

        val_loss_trace.append(val_loss); val_acc_trace.append(val_acc)  # Append validation accuracy for current epoch
        scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0 # reset
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return (loss_trace, epoch_loss_trace, val_loss_trace, val_acc_trace)

def evaluateConnCNN_model(model, test_loader, criterion):
    model.eval()
    
    num_correct = 0; connect_loss = 0; pred_list = []; label_list = []; w_list = []; sInfo_list = []
    with torch.no_grad():
        for ccgs, labels, weights, s_info in test_loader:

            connectivity_out = model(ccgs)

            connectivity_loss = criterion(connectivity_out, labels)
            connect_loss += connectivity_loss.item() * ccgs.size(0)
            
            # predicted_connectivity = (torch.sigmoid(connectivity_out) > 0.51).float()
            predicted_connectivity = (connectivity_out > 0.51).float()
            num_correct += (predicted_connectivity == labels).sum().item()  # correct ones

            pred_list.extend(predicted_connectivity.tolist())
            label_list.extend(labels.squeeze(1).tolist())
            w_list.extend(weights.squeeze(1).tolist())
            sInfo_list += list(s_info)


    avg_connect_loss = connect_loss / len(test_loader.dataset)
    accu = num_correct / len(test_loader.dataset) * 100

    return(avg_connect_loss, accu, np.array(pred_list).flatten(), np.array(label_list).flatten(), np.array(w_list).flatten(), np.array(sInfo_list) )

def evaluateConnCNN_wConfi_model(model, test_loader, criterion):
    model.eval()
    
    num_correct = 0; connect_loss = 0; pred_list = []; label_list = []; w_list = []; sInfo_list = []; confi_list = []
    with torch.no_grad():
        for ccgs, labels, weights, s_info in test_loader:

            # model predict - logits
            connectivity_out = model(ccgs)  # logits

            # calculate loss
            connectivity_loss = criterion(connectivity_out, labels)
            connect_loss += connectivity_loss.item() * ccgs.size(0)

            # logits -> confidence level in [0, 1]
            confidence_scores = torch.sigmoid(connectivity_out)

            # binary 
            predicted_connectivity = (confidence_scores > 0.5).float()
            num_correct += (predicted_connectivity == labels).sum().item()  # correct ones

            # save
            confi_list.extend(confidence_scores.squeeze(1).tolist())
            pred_list.extend(predicted_connectivity.squeeze(1).tolist())
            label_list.extend(labels.squeeze(1).tolist())
            w_list.extend(weights.squeeze(1).tolist())
            sInfo_list += list(s_info)


    avg_connect_loss = connect_loss / len(test_loader.dataset)
    accu = num_correct / len(test_loader.dataset) * 100

    return(avg_connect_loss, accu, 
           np.array(pred_list).flatten(), np.array(confi_list).flatten(), 
           np.array(label_list).flatten(), np.array(w_list).flatten(), np.array(sInfo_list) )




class WeightCNN(nn.Module): # original
    
    def __init__(self, input_dim):
        super (WeightCNN, self).__init__()
        
        conv1_layer=4; kernel1_size=70
        conv2_layer=2; kernel2_size=35

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_layer, 
                               kernel_size=kernel1_size, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(conv1_layer)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv2 = nn.Conv1d(in_channels=conv1_layer, out_channels=conv2_layer, 
                               kernel_size=kernel2_size, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(conv2_layer)


        self.global_max_pool = nn.AdaptiveMaxPool1d(1)  # Reduces sequence to a single vector
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # This will help generalize better

        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.fc = nn.Linear(conv2_layer, 1)
        

    def forward(self, x):
        
        if x.dim() == 2: 
            x = x.unsqueeze(1)
        
        x = self.bn1(self.conv1(x))
        x = self.tanh(x)
        x = self.bn2(self.conv2(x))
        x = self.tanh(x)
        x = self.global_avg_pool(x); 
        x = x.squeeze(-1) # torch.Size([32, 128])
        
        x = self.dropout(x)

        weight = self.fc(x) # torch.Size([32, 1])
        
        return weight
    
def trainWeightCNN_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, patience=3):
    model.train()
    
    best_val_loss = float('inf'); counter=0; loss_trace = []; epoch_loss_trace = []; val_loss_trace = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for ccgs, labels, weights, _ in train_loader:
            optimizer.zero_grad()
            
            weight_out = model(ccgs)
            weight_loss = criterion(weight_out, weights)   
            
            # weighted MSE
            label_mask = (weights > 0).float(); weight_factor = 5.0
            loss = (weight_loss * (1 + weight_factor * label_mask)).mean()

            
            #loss = weight_loss
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item() * ccgs.size(0)
            running_loss += batch_loss
            loss_trace.append(batch_loss)
            
    
        # average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_loss_trace.append(epoch_loss)
        
        # dynamically adjust learning rate
        val_loss,_,_,_,_,_= evaluateWeightCNN_model(model, val_loader, criterion)
        scheduler.step(val_loss)
        val_loss_trace.append(val_loss)
        
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0 # reset
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
    return (loss_trace, epoch_loss_trace, val_loss_trace)
       
def evaluateWeightCNN_model(model, test_loader, criterion):
    
    z0_loss=[]; zn0_loss=[]; running_loss = 0.0; predW_list = []; gtW_list = [];  sInfo_list = []
    
    model.eval()
    with torch.no_grad(): # Disable gradient calculation
        for ccgs, labels, weights, s_info in test_loader:
            
            weight_out = model(ccgs)
            
            # track weight loss for label-0/1 separately
            zero_mask = (weights.squeeze(1) == 0)
            nonzero_mask = ~zero_mask
            if zero_mask.any():
                zero_loss = criterion(weight_out[zero_mask].flatten(), weights.squeeze(1)[zero_mask]).item()
            else:
                zero_loss = 0.0

            if nonzero_mask.any():
                nonzero_loss = criterion(weight_out[nonzero_mask].flatten(), weights.squeeze(1)[nonzero_mask]).item()
            else:
                nonzero_loss = 0.0
            z0_loss.append(zero_loss);zn0_loss.append(nonzero_loss)
            
            
            weight_loss = criterion(weight_out.flatten(), weights.squeeze(1))
            running_loss += weight_loss.item() * ccgs.size(0)  
            
            predW_list.extend(weight_out.squeeze(1).tolist())
            gtW_list.extend(weights.squeeze(1).tolist())

            sInfo_list += list(s_info)
            
    avg_wegiht_loss = running_loss/len(test_loader.dataset)
    avg_label0_loss = np.mean(z0_loss); avg_label1_loss = np.mean(zn0_loss)
    
    return (avg_wegiht_loss, np.array(predW_list), np.array(gtW_list), avg_label0_loss, avg_label1_loss, np.array(sInfo_list))

class HuberLossWithWeight(nn.Module):
    def __init__(self, delta=1.0, weight_decay=2.0, threshold=0.1):
        super(HuberLossWithWeight, self).__init__()
        self.delta = delta
        self.weight_decay = weight_decay
        self.threshold = threshold
    
    def forward(self, y_true, y_pred):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        
        # Huber loss formula
        loss = torch.where(abs_error <= self.delta, 0.5 * (error ** 2), self.delta * (abs_error - 0.5 * self.delta))
        
        # Exponential weight based on proximity to zero
        weight = torch.exp(-self.weight_decay * torch.abs(y_true))  # Exponentially higher weight near 0
        weight = torch.where(torch.abs(y_true) < self.threshold, weight, torch.tensor(1.0))
        
        # Apply weight to the loss
        weighted_loss = loss * weight
        
        return weighted_loss.mean()



class ConnCNN_DualInput(nn.Module):
    def __init__(self, input_dim=101):
        super(ConnCNN_DualInput, self).__init__()

        conv1_layer=4; kernel1_size=70
        conv2_layer=2; kernel2_size=35
        
        # raw branch
        self.raw_conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_layer, kernel_size=kernel1_size, stride=1, padding=2)
        self.raw_bn1 = nn.BatchNorm1d(conv1_layer)
        self.raw_conv2 = nn.Conv1d(in_channels=conv1_layer, out_channels=conv2_layer, kernel_size=kernel2_size, stride=1, padding=2)
        self.raw_bn2 = nn.BatchNorm1d(conv2_layer)

        # scaled branch
        self.scale_conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_layer, kernel_size=kernel1_size, stride=1, padding=2)
        self.scale_bn1 = nn.BatchNorm1d(conv1_layer)
        self.scale_conv2 = nn.Conv1d(in_channels=conv1_layer, out_channels=conv2_layer, kernel_size=kernel2_size, stride=1, padding=2)
        self.scale_bn2 = nn.BatchNorm1d(conv2_layer)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # FC layer after concatenation: input dim = 2 (raw) + 2 (scaled)
        self.fc = nn.Linear(4, 1)     # 2 features from each branch
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        """
        x: Tensor of shape [B, 2, 101]
        x[:, 0, :] is raw CCG
        x[:, 1, :] is scaled CCG
        """

        raw = x[:, 0, :].unsqueeze(1)    # [B, 1, 101]
        scale = x[:, 1, :].unsqueeze(1)  # [B, 1, 101]

        # raw branch
        raw = self.raw_bn1(self.raw_conv1(raw))
        raw = self.tanh(raw)
        raw = self.raw_bn2(self.raw_conv2(raw))
        raw = self.tanh(raw)
        raw = self.global_avg_pool(raw).squeeze(-1)  # [B, 2]

        # scaled branch
        scale = self.scale_bn1(self.scale_conv1(scale))
        scale = self.tanh(scale)
        scale = self.scale_bn2(self.scale_conv2(scale))
        scale = self.tanh(scale)
        scale = self.global_avg_pool(scale).squeeze(-1)  # [B, 2]

        # concatenate and pass through shared FC head
        x = torch.cat([raw, scale], dim=1)  # [B, 4]
        x = self.dropout(x)
        out = self.fc(x)                    # [B, 1]
        
        return out


class WeightCNN_DualInput(nn.Module):
    def __init__(self, input_dim=101):
        super(WeightCNN_DualInput, self).__init__()
        
        conv1_layer=4; kernel1_size=70
        conv2_layer=2; kernel2_size=35

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # raw branch
        self.raw_conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_layer, kernel_size=kernel1_size, stride=1, padding=2)
        self.raw_bn1 = nn.BatchNorm1d(conv1_layer)
        self.raw_conv2 = nn.Conv1d(in_channels=conv1_layer, out_channels=conv2_layer, kernel_size=kernel2_size, stride=1, padding=2)
        self.raw_bn2 = nn.BatchNorm1d(conv2_layer)

        # scaled branch
        self.scale_conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_layer, kernel_size=kernel1_size, stride=1, padding=2)
        self.scale_bn1 = nn.BatchNorm1d(conv1_layer)
        self.scale_conv2 = nn.Conv1d(in_channels=conv1_layer, out_channels=conv2_layer, kernel_size=kernel2_size, stride=1, padding=2)
        self.scale_bn2 = nn.BatchNorm1d(conv2_layer)

        # FC head after concatenating pooled raw + scale → (2 + 2) = 4 features
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        """
        x: [B, 2, 101] where
           x[:, 0, :] → raw CCG
           x[:, 1, :] → scaled CCG
        """
        raw = x[:, 0, :].unsqueeze(1)    # [B, 1, 101]
        scale = x[:, 1, :].unsqueeze(1)  # [B, 1, 101]

        # raw branch
        raw = self.tanh(self.raw_bn1(self.raw_conv1(raw)))
        raw = self.tanh(self.raw_bn2(self.raw_conv2(raw)))
        raw = self.global_avg_pool(raw).squeeze(-1)  # [B, 2]

        # scaled branch
        scale = self.tanh(self.scale_bn1(self.scale_conv1(scale)))
        scale = self.tanh(self.scale_bn2(self.scale_conv2(scale)))
        scale = self.global_avg_pool(scale).squeeze(-1)  # [B, 2]

        # concatenate, dropout, predict weight
        x = torch.cat([raw, scale], dim=1)  # [B, 4]
        x = self.dropout(x)
        weight = self.fc(x)   # [B, 1]

        return weight







# CNN analysis =============================================================================================
def to_tensor_1d(x, device):
    # array to tensor
    t = torch.as_tensor(x, dtype=torch.float32, device=device).squeeze()  # (L,)
    return t

# Saliency Map ---------------------
def scale_values(input_array):
    ''' Min-Max scaling to [0, 1] '''
    if type(input_array) == torch.Tensor:
        input_array = input_array.detach().cpu().numpy()

    input_min, input_max = np.min(input_array), np.max(input_array)
    input_scaled = (input_array - input_min) / (input_max - input_min)

    return input_scaled

def compute_saliency_map(model, input_sample, target_index=None, ifScale=True):
    '''
    Compute a saliency map for a single 1D-CCG input sample - gradient magnitude the output with respect to the input
    Higher gradient magnitude - import bins

    Args:
        model           : nn.Module, your trained CNN.
        input_sample    : torch.Tensor, shape (C, T)—your CCG channels × timebins.
        target_index    : int or None. 
                          For classification: index of the logit to explain (e.g. 0 for “connected”). 
                          If None, assumes output is a single scalar and uses that.
        normalize       : bool. If True, scales saliency to [0,1].

    Returns:
        saliency_map    : np.ndarray, shape (C, T), absolute-gradient saliency.
    '''
    
    
    model.eval()

    # Make a leaf tensor that requires grad
    x = input_sample.clone().detach().unsqueeze(0)  # → shape (1, C, T)
    x.requires_grad_(True)

    # Forward pass
    out = model(x)  # → shape (1,) or (1, num_classes)

    # Pick the scalar to backprop through
    if target_index is not None:
        # e.g. for multi-class use out[0, target_index]
        score = out[0, target_index]
    else:
        # assume out is shape (1,) or (1,1)
        score = out.view(-1)[0]

    # Zero gradients
    model.zero_grad()

    # Backward pass
    score.backward()

    # Absolute gradient w.r.t. the input
    saliency = x.grad.abs().squeeze(0).detach().cpu().numpy()  # → shape (C, T)

    # Optionally normalize each channel to [0,1]
    if ifScale:
        return scale_values(saliency)
    else:
        return  saliency



# Grad-CAM ------------------------
def grad_cam_1d(model, input_data, target_layer, class_idx=None):
    model.eval()
    input_data = input_data.unsqueeze(0)
    # register hook to capture activations and gradients
    activations = []; gradients = []
    def save_activation(module, input, output):
        activations.append(output)
    def save_gradient(module, grad_in, grad_out):
        #print(grad_out)
        gradients.append(grad_out[0])
    
    # register hooks on the target layer (e.g., conv2 layer)
    target_layer.register_forward_hook(save_activation)
    target_layer.register_backward_hook(save_gradient)
    

    output = model(input_data)
    if class_idx is None: # if class_idx is None, use the class with the highest score
        class_idx = torch.argmax(output)
    
    model.zero_grad()     # zero gradients, backpropagate
    output[0, class_idx].backward()
    
    # get the activation and gradient
    activation = activations[0].detach().cpu().numpy()
    gradient = gradients[0]
    
    # compute the importance weights for each filter (global average pooling of gradients)
    weights = torch.mean(gradient, dim=(2))  # Global average of gradients over the height (time)
    
    # compute Grad-CAM
    grad_cam_map = torch.zeros(activation.shape[2], dtype=torch.float32)
    for i in range(activation.shape[1]):  # Loop through all channels (filters)
        grad_cam_map += weights[0, i] * activation[0, i, :]
    
    # normalize the Grad-CAM map
    grad_cam_map = F.relu(grad_cam_map)
    grad_cam_map -= grad_cam_map.min()
    grad_cam_map /= (grad_cam_map.max() + 1e-8)
    
    return grad_cam_map

def upsample_cam(cam, target_length):
    '''
    cam: 1D numpy or torch tensor of length L' (e.g. 6)
    target_length: e.g. 101
    returns: 1D numpy array of length target_length
    '''

    if not isinstance(cam, torch.Tensor):
        cam = torch.tensor(cam)
    cam = cam.unsqueeze(0).unsqueeze(0)
    cam_up = F.interpolate(cam, size=target_length, mode='linear', align_corners=False)
    
    return cam_up.squeeze().cpu().numpy()


# Layer-wise activatons -----------
def cal_feature_maps(model, data):

    '''
    this function is used in `ML_Models/2-0-CNN_Connectivity.ipynb`, `ML_Models/6-CNN_Weight.ipynb`
    obtain the feature maps from a 1D CNN for a single ccg input
    feature map is the output from each convolution layer:
        - feature map from self.conv1: filter scan through ccg resulting in shorter length
            - e.g., self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=2, padding=2)
            - each kernal has lengh of 5 and 4 filters in total
            - it outputs 4 feature maps, each has length of 25
            
        - feature map from self.conv2: filter scan through feature maps from self.conv1
            - e.g., self.conv2 = nn.Conv1d(in_channels=conv1_layer, out_channels=8, kernel_size=3, stride=1, padding=2)
            - 8 filters in total and each filter has length of 3
            - 8 filters scan through 4 channels (feature maps from conv1) and for each channel
                - 8 output and each output has lengh of 25
                - there's a weight kernal for each channel, and the scanning reuslts are weighted-sum+bias
            - so total output (feature maps) in shape: [batch_size, 8, 25]
    '''
    
    model.eval()

    # Pass data through the model and get the feature maps from the convolutional layers
    with torch.no_grad():
        x = data.unsqueeze(0)  # Add batch dimension if necessary
        activations = []

        # Define a hook function to capture the activations of the first convolutional layer (conv1)
        def hook_fn_conv1(module, input, output):
            activations.append(output)  # Save the output of conv1

        # Define a hook function to capture the activations of the second convolutional layer (conv2)
        def hook_fn_conv2(module, input, output):
            activations.append(output)  # Save the output of conv2

        # Register hooks on the convolutional layers
        hook_conv1 = model.conv1.register_forward_hook(hook_fn_conv1)
        hook_conv2 = model.conv2.register_forward_hook(hook_fn_conv2)

        # Forward pass
        _ = model(x)  # This triggers both hooks and stores activations

        # Remove the hooks after the forward pass
        hook_conv1.remove()
        hook_conv2.remove()

    # Get the feature maps (activations) for both convolutional layers
    feature_maps_conv1 = activations[0].squeeze(0)  # Remove batch dimension for conv1
    feature_maps_conv2 = activations[1].squeeze(0)  # Remove batch dimension for conv2
    
    activation_values_1 = feature_maps_conv1.flatten().cpu().numpy()  # Flatten activations from conv1
    activation_values_2 = feature_maps_conv2.flatten().cpu().numpy()  # Flatten activations from conv1
    
    return(feature_maps_conv1, feature_maps_conv2, activation_values_1, activation_values_2)

def fisher_discriminant_ratio(features, labels):
    class0 = features[labels == 0]
    class1 = features[labels == 1]
    
    mean0 = np.mean(class0, axis=0)
    mean1 = np.mean(class1, axis=0)
    
    var0 = np.var(class0, axis=0)
    var1 = np.var(class1, axis=0)
    
    numerator = np.sum((mean0 - mean1) ** 2)
    denominator = np.sum(var0 + var1) + 1e-9  # add epsilon to avoid divide-by-zero
    
    return round(numerator / denominator, 5)

def compute_silhouette(features, labels):
    try:
        return round(silhouette_score(features, labels), 5)
    except:
        return np.nan  # in case only one label or other issues

def linear_classification_accuracy(features, labels):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(features, labels)
    predictions = clf.predict(features)
    return round(accuracy_score(labels, predictions), 5)

def evaluate_discriminability(activation_list, labels):
    features = np.array([a.flatten() for a in activation_list])
    labels = np.array(labels)

    fdr = fisher_discriminant_ratio(features, labels)
    sil_score = compute_silhouette(features, labels)
    lin_acc = linear_classification_accuracy(features, labels)

    return { "fisher_ratio": fdr, "silhouette_score": sil_score, "linearClassification_accu": lin_acc}


# filters -------------------------
def plot_conv1_filters(model, ifSave=False, savePath=None, filename=None):
    # Extract Conv1 weights
    conv1_weights = model.conv1.weight.data.cpu().numpy()  # Shape: (out_channels=4, in_channels=1, kernel_size)
    
    conv1_weights = model.conv1.weight.data.cpu().numpy().squeeze(1)  # shape: (4, 70)

    plt.figure(figsize=(8, 3))
    sns.heatmap(conv1_weights, cmap='coolwarm', cbar=True, xticklabels=False, yticklabels=[f'F{i+1}' for i in range(4)])
    plt.tight_layout(pad=0)
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename), bbox_inches="tight", pad_inches=0, )
        plt.close()
    else:
        plt.show()

def plot_conv2_filters(model, ifSave=False, savePath=None, filename=None):
    """
    Plots Conv2 filters as 2D heatmaps (in_channels x kernel_size).
    """
    weights = model.conv2.weight.data.cpu().numpy()  # shape: (out_channels, in_channels, kernel_size)
    out_channels, in_channels, kernel_size = weights.shape

    fig, axes = plt.subplots(1, out_channels, figsize=(6*out_channels, 3))
    for i in range(out_channels):
        ax = axes[i] if out_channels > 1 else axes
        sns.heatmap(weights[i], cmap='coolwarm', cbar=True, ax=ax)
    plt.tight_layout(pad=0)
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename), bbox_inches="tight", pad_inches=0, )
        plt.close()
    else:
        plt.show()

def calculate_degradation(original_mse, new_mse):
    return ((new_mse - original_mse) / original_mse) * 100
    
def ablate_filter(model, conv_layer_idx, filter_idx):
    if conv_layer_idx == 1:
        model_copy = copy.deepcopy(model)
        model_copy.conv1.weight.data[filter_idx] = 0
        model_copy.conv1.bias.data[filter_idx] = 0
    elif conv_layer_idx == 2:
        model_copy = copy.deepcopy(model)
        model_copy.conv2.weight.data[filter_idx] = 0
        model_copy.conv2.bias.data[filter_idx] = 0
    return model_copy

def restore_filter(model, layer_name, filter_idx, original_filter):
    '''restore the original filter after it has been ablated'''
    conv_layer = getattr(model, layer_name)
    conv_layer.weight.data[filter_idx] = original_filter

def ablate_evaluate_W_model(model, test_loader, loss_fn=F.mse_loss):
    '''evaluate the model on the test set and return the average loss'''
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for inputs, _, targets, _ in test_loader:
            output = model(inputs)  # Forward pass
            total_loss += loss_fn(output, targets).item()  # Compute the loss
            num_batches += 1
    return total_loss / num_batches  # Return average loss

def ablate_evaluate_C_model(model, test_loader, loss_fn=nn.BCEWithLogitsLoss()):
    '''evaluate the model on the test set and return the average loss'''
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for inputs, targets, _, _ in test_loader:
            output = model(inputs)  # Forward pass
            total_loss += loss_fn(output, targets).item()  # Compute the loss
            num_batches += 1
    return total_loss / num_batches

def ablation_study(model, test_loader, modelType='W'):   
    ''' 
    perform the ablation study for a specific filter in conv1
    evaluate the model's performance (MSE loss) before and after the ablation
    '''
    if modelType=='W':
        baseline_loss = ablate_evaluate_W_model(model, test_loader)
    elif modelType=='C':
        baseline_loss = ablate_evaluate_C_model(model, test_loader)
    else:
        raise ValueError('CNN modelType must not right. ')
    print(f"Baseline Loss (before ablation): {baseline_loss}")

    ablation_results = {}
    for filter_idx in range(model.conv1.out_channels):
        ablated_model = ablate_filter(model, 1, filter_idx)  # Ablating filter in conv1
        if modelType=='W':
            loss = ablate_evaluate_W_model(ablated_model, test_loader)
        elif modelType=='C':
            loss = ablate_evaluate_C_model(ablated_model, test_loader)
        else:
            raise ValueError('CNN modelType must not right. ')
        performance_degradation = calculate_degradation(baseline_loss, loss)
        ablation_results[f"conv1_filter_{filter_idx}"] = {
                                    "Loss": round(loss, 5),
                                    "Performance Degradation (%)": round(performance_degradation, 5)}

    # loop over 2nd convolutional layer filters
    for filter_idx in range(model.conv2.out_channels):
        ablated_model = ablate_filter(model, 2, filter_idx)  # Ablating filter in conv2
        if modelType=='W':
            loss = ablate_evaluate_W_model(ablated_model, test_loader)
        elif modelType=='C':
            loss = ablate_evaluate_C_model(ablated_model, test_loader)
        else:
            raise ValueError('CNN modelType must not right. ')
        performance_degradation = calculate_degradation(baseline_loss, loss)
        ablation_results[f"conv2_filter_{filter_idx}"] = {
                                    "Loss": round(loss, 5),
                                    "Performance Degradation (%)": round(performance_degradation, 5)}
    
    return baseline_loss, ablation_results  






# CCGs realted  ==========================================================================================
def read_ccg(file_path, file_name):

    CCH_df = pd.read_csv(os.path.join(file_path, file_name)).iloc[:, 1:]
    ccg_data = CCH_df[0:-1].T.to_numpy()
    
    weights= CCH_df.iloc[-1:].to_numpy()[0]
    presence = np.array([0 if l==0 else 1 for l in weights])
    sample_info = np.array(CCH_df.columns.tolist())

    return ccg_data, weights, presence, sample_info

def ccg_indicators(ccg, connection_type='exc', smoothSigma=0.5, bin_size=1, peak_window_ms=10, ifVerbose=False, ifPlot=False, timebins=None, figsize=(8,2), barColor='lightslategrey', spanColor='firebrick', ifSave=True, savePath=None, filename=None):
    '''
    Calculate signal indicators for CCGs.
    For excitatory connections (peaks) and inhibitory connections (dips), adjusts indicators accordingly.
    
    Returns: a dictionary with standardized keys:
        - peak_height or dip_depth
        - peak_to_noise or dip_to_noise
        - peak_halfMax_width or dip_halfMax_width
        - peak_lag or dip_lag
        - entropy
        - temporal_span
    '''
    
    
    ccg = np.array(ccg, dtype=np.float32)
    N = len(ccg); center = N // 2
    baseline = np.mean(np.concatenate([ccg[:10], ccg[-10:]]))


    # search for peak/dip -----------------------------------------
    # restrict to [-peak_window_ms,  peak_window_ms] range around the center, e.g., -10ms ~ 10ms
    search_bins = int(peak_window_ms / bin_size)
    search_region = ccg[center-search_bins : center+search_bins+1]
    if connection_type == 'exc':
        peak_idx_rel = np.argmax(search_region)
        peak_idx = center - search_bins + peak_idx_rel
        peak_val = ccg[peak_idx]
        direction = 'peak'
    elif connection_type == 'inh':
        peak_idx_rel = np.argmin(search_region)
        peak_idx = center - search_bins + peak_idx_rel
        peak_val = ccg[peak_idx]
        direction = 'dip'
    else:
        raise ValueError("connection_type must be 'exc' or 'inh'")
    
    peak_lag = (peak_idx - center) * bin_size


    # half-max width -----------------------------------------------
    half_val = peak_val / 2 if connection_type == 'exc' else (peak_val + baseline) / 2
    left, right = peak_idx, peak_idx
    while left > 0 and ((ccg[left] > half_val) if connection_type == 'exc' else (ccg[left] < half_val)):
        left -= 1
    while right < len(ccg) - 1 and ((ccg[right] > half_val) if connection_type == 'exc' else (ccg[right] < half_val)):
        right += 1
    peak_width = (right - left) * bin_size


    # noise estimation from tails ----------------------------------
    tail_bins = np.r_[np.arange(25), np.arange(len(ccg) - 25, len(ccg))]
    noise_floor = np.mean(ccg[tail_bins])
    noise_std = np.std(ccg[tail_bins])
    peak_to_noise = ((peak_val - noise_floor) / (noise_std + 1e-10)) if connection_type == 'exc' else ((noise_floor - peak_val) / (noise_std + 1e-10))


    # smooth for entropy and temporal span --------------------------
    ccg_smooth = gaussian_filter1d(ccg, sigma=smoothSigma)
    ccg_prob = ccg_smooth / (np.sum(ccg_smooth) + 1e-10)
    ccg_entropy = entropy(ccg_prob, base=2)
    ccg_entropy_norm = ccg_entropy / np.log2(N)

    # KL Divergence (+- 10ms window) ---------------------------------
    window_bins = int(peak_window_ms / bin_size)
    kl_window = ccg[center - window_bins:center + window_bins + 1]
    P = kl_window / (np.sum(kl_window) + 1e-10)
    U = np.ones_like(P) / len(P)
    kl_div = np.sum(rel_entr(P, U)) / np.log(2)  # in bits

    # entropy (+- 10ms window) -----------------------------------------
    window_entropy = entropy(P, base=2)
    window_entropy_norm = window_entropy / np.log2(len(P))


    # temporal span based on threshold
    thresh = noise_floor + 2 * noise_std if connection_type == 'exc' else noise_floor - 2 * noise_std
    left_span, right_span = peak_idx, peak_idx
    while left_span > 0 and ((ccg_smooth[left_span] > thresh) if connection_type == 'exc' else (ccg_smooth[left_span] < thresh)):
        left_span -= 1
    while right_span < len(ccg_smooth) - 1 and ((ccg_smooth[right_span] > thresh) if connection_type == 'exc' else (ccg_smooth[right_span] < thresh)):
        right_span += 1
    temporal_span = (right_span - left_span) * bin_size

    if ifVerbose:
        print(f"{direction} occurs at bin {peak_idx} = {peak_val} with lag = {peak_lag}")
        print(f"{direction} drops to half in {peak_width} bins")
        print(f"Tails: mean = {round(noise_floor, 3)}, std = {round(noise_std, 3)}, {direction}_to_noise = {round(peak_to_noise, 3)}")
        print(f"Temporal span above threshold ({thresh:.2f}) = {temporal_span} bins around center")
        print(f"Entropy = {round(ccg_entropy, 3)}")
        print(f"Normalized entropy = {ccg_entropy_norm:.3f}")
        print(f"Normalized entropy (±{peak_window_ms} ms) = {window_entropy_norm:.3f}")
        print(f"KL divergence (±{peak_window_ms} ms) = {kl_div:.3f}")
        

    if ifPlot:
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].bar(timebins, ccg, color=barColor, width=1); ax[0].set_title('raw')
        ax[1].bar(timebins, ccg_smooth, color=barColor, width=1); ax[1].set_title('smoothed')
        ax[1].bar(timebins[left_span:right_span], ccg_smooth[left_span:right_span], 
                  color=spanColor, width=1, alpha=0.4, label='temporal span'); ax[1].legend(loc=1)
        ax[2].bar(timebins[center-window_bins : center+window_bins+1], kl_window, color='g', width=1);ax[2].set_title('+-10 focus window')
        plt.tight_layout()
        if ifSave:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(os.path.join(savePath, filename))
            plt.close()
        else:
            plt.show()

    # Return unified dictionary with consistent key naming
    result = {
        f"{direction}_height" if direction == 'peak' else "dip_depth": peak_val,
        f"{direction}_lag": peak_lag,
        f"{direction}_halfMax_width": peak_width,
        f"{direction}_to_noise": peak_to_noise,
        "temporal_span": temporal_span,
        "entropy": ccg_entropy,
        "norm_entropy": ccg_entropy_norm,
        'norm_entropy_window': window_entropy_norm, 
        "kl_divergence_window": kl_div 
    }

    return result

def scale_ccg_baseline(ccg, baseline_bins=10):
    '''
    Scale a CCG by centering on baseline and dividing by max absolute deviation.
    
    Parameters:
        - ccg (array-like): The CCG values (1D array).
        - baseline_bins (int): # of bins from start and end to compute the baseline average.
    
    Returns:
        scaled_ccg (np.ndarray): Scaled CCG with baseline-centered and peak normalized.
    '''
    
    ccg = np.array(ccg, dtype=np.float32)
    
    # baseline as average of first and last baseline_bins
    baseline = np.mean(np.concatenate([ccg[:baseline_bins], ccg[-baseline_bins:]]))
    
    # Center the CCG around the baseline
    shift_ccg = ccg - baseline
    
    # normalize by max absolute value
    ccg_max = np.max(np.abs(shift_ccg))
    scaled_ccg = shift_ccg / ccg_max if ccg_max != 0 else shift_ccg
    
    return scaled_ccg

def scale_ccg_baseline_batch(ccg_array, baseline_bins=10):
    '''
    a batch 
    '''
    ccg_array = np.array(ccg_array, dtype=np.float32)
    scaled_array = np.zeros_like(ccg_array)

    for i in range(ccg_array.shape[0]):
        scaled_ccg = scale_ccg_baseline( ccg_array[i], baseline_bins=baseline_bins)
        scaled_array[i] = scaled_ccg
    
    return scaled_array

def jitter_spikes(spk_times, w, T):
    """Uniform jitter in [-w, +w]; keep within [0, T]."""
    if spk_times.size == 0:
        return spk_times.copy()
    jit = np.random.uniform(-w, +w, size=spk_times.size)
    s = spk_times + jit
    return s[(s >= 0.0) & (s <= T)]











# Others  ================================================================================================
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)                # Python built-in
    np.random.seed(seed)            # NumPy
    torch.manual_seed(seed)         # PyTorch (CPU)
    torch.cuda.manual_seed(seed)    # PyTorch (current GPU)
    torch.cuda.manual_seed_all(seed) # All GPUs (if using multi-GPU)
    
    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def jaccard_index(set1, set2):
    '''
    Jaccard index (Jaccard similarity coefficient) to measure overlap between two sets
    for 2 sets of same length, a Jaccard < 0.6 indicates that less than 60^ of their elements are shared, relatively low overlap
    '''
    set1, set2 = set(set1), set(set2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def compute_aps_best_mcc(y_true, y_score):
    """
    Returns
    {"APS": float,
     "best_MCC": float,
     "best_threshold": float,
     "MCC": float,                 # MCC at fixed threshold = 0.5
     "TP_str": str                 # "tp/num_positives", e.g., "16/26"
    }

    Notes
        - The fixed-threshold MCC uses threshold = 0.5. Make sure y_score is a probability in [0,1]
        - The bset MCC is calculated over thresholds - 
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Drop NaNs / infs
    mask = np.isfinite(y_score)
    y_true = y_true[mask]; y_score = y_score[mask]

    # APS (threshold-free)
    aps = float(average_precision_score(y_true, y_score))

    # Best MCC over thresholds (skip degenerate all-0/all-1 predictions)
    order = np.argsort(-y_score)
    s = y_score[order]; t = y_true[order]
    uniq = np.unique(s); best_mcc, best_thr = -1.0, None
    if len(uniq) == 1:                       # all scores identical -> only one prediction possible
        thr = float(uniq[0])
        y_pred = (y_score >= thr).astype(int)
        best_mcc = float(matthews_corrcoef(y_true, y_pred))
        best_thr = thr
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0  # decision points where predictions change
        for thr in mids:
            y_pred = (y_score >= thr).astype(int)
            p = y_pred.sum()
            if p == 0 or p == len(y_pred):  # skip degenerate all-0/all-1
                continue
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_mcc:
                best_mcc, best_thr = float(mcc), float(thr)
        
        if best_thr is None:                # Fallback if all mids degenerate (rare)
            thr = 0.5 if (y_score.min() >= 0.0 and y_score.max() <= 1.0) else float(np.median(uniq))
            y_pred = (y_score >= thr).astype(int)
            best_mcc = float(matthews_corrcoef(y_true, y_pred))
            best_thr = float(thr)


    # Plain MCC at threshold = 0.5 (assumes scores are probabilities)
    fixed_thr = 0.5
    y_pred_fixed = (y_score >= fixed_thr).astype(int)
    mcc_fixed = float(matthews_corrcoef(y_true, y_pred_fixed))

    # TP string: "tp/num_positives"
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_fixed, labels=[0, 1]).ravel()
    total = int(len(y_true))
    pos_total = int((y_true == 1).sum())
    neg_total = int((y_true == 0).sum())

    # Overall accuracy
    overall_correct = int(tp + tn)
    acc_overall = overall_correct / total if total > 0 else float("nan")
    acc_overall_str = f"{overall_correct}/{total}"

    # Per-class accuracy (a.k.a. recall per class)
    acc_pos = tp / pos_total if pos_total > 0 else float("nan")
    acc_pos_str = f"{int(tp)}/{pos_total}"

    acc_neg = tn / neg_total if neg_total > 0 else float("nan")
    acc_neg_str = f"{int(tn)}/{neg_total}"

    # TP string for convenience (positives)
    tp_str = f"{int(tp)}/{pos_total}"


    return {"APS": aps,
            "best_MCC": round(best_mcc, 3), "best_threshold": best_thr,  
            "MCC": round(mcc_fixed, 3),
            "TP_str": tp_str,

            "acc_overall_str": acc_overall_str,
            "acc_overall": round(acc_overall, 3),

            "acc_unconn_str": acc_neg_str,
            "acc_unconn": round(acc_neg, 3),

            "acc_conn_str": acc_pos_str,
            "acc_conn": round(acc_pos, 3), }
















# MLP analysis =============================================================================================
def extract_MLP_hidden_activations(model, dataloader, activationMethod='ReLU'):
    '''
    extracts activations from a specified hidden layer during forward pass
    activations are the values that neurons output after processing input data
    '''
    model.eval()  
    fc1_activations=[]; fc2_activations=[]; label_list = []; weight_list = []; sInfo_list = []
    fc1_conn_activs=[]; fc1_unconn_activs=[]; fc2_conn_activs=[]; fc2_unconn_activs=[]

    with torch.no_grad():
        for ccgs, labels, weights, s_info in dataloader:
            x = ccgs  
            
            if activationMethod == 'ReLU':
                x = model.relu(model.fc1(x))
            elif activationMethod == 'Tanh':
                x = model.tanh(model.fc1(x))
            fc1_activation = x.detach().cpu().numpy()  # Extract activations
            
            if activationMethod == 'ReLU':
                x = model.relu(model.fc2(x))
            elif activationMethod == 'Tanh':
                x = model.tanh(model.fc2(x))
            fc2_activation = x.detach().cpu().numpy()

            conns = np.where(labels!=0)[0]; unconns = np.where(labels==0)[0]; 
            
            fc1_activations.append(fc1_activation)
            fc1_conn_activs.append(fc1_activation[conns]); fc1_unconn_activs.append(fc1_activation[unconns])
            
            fc2_activations.append(fc2_activation)
            fc2_conn_activs.append(fc2_activation[conns]); fc2_unconn_activs.append(fc2_activation[unconns])
            
            
            label_list.append(labels.cpu().numpy())
            weight_list.append(weights.cpu().numpy())
            sInfo_list+=list(s_info)
    
    fc1_acs=np.vstack(fc1_activations); fc1_conn_acs=np.vstack(fc1_conn_activs); fc1_unconn_acs=np.vstack(fc1_unconn_activs)
    fc2_acs=np.vstack(fc2_activations); fc2_conn_acs=np.vstack(fc2_conn_activs); fc2_unconn_acs=np.vstack(fc2_unconn_activs)
    
    return fc1_acs, fc1_conn_acs, fc1_unconn_acs, fc2_acs, fc2_conn_acs, fc2_unconn_acs, np.concatenate(label_list), np.concatenate(weight_list), np.array(sInfo_list)





