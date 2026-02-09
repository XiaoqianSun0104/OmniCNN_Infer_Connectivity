# 3_genCCG.py
# Author: Xiaoqian Sun, 03/16/2025
# Fucntion: generate pairwise CCG using 1 spkTimes df



# Import Packages
#=================================================================================================================
import os
import time
import h5py
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

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

import warnings
warnings.filterwarnings('ignore')

# argumetns
#=================================================================================================================
parser = argparse.ArgumentParser("Static Network Simulation CCG Generation")
parser.add_argument('--dataFolderName', default='C_Data', type=str)
parser.add_argument('--segFolderName', default='static_network', type=str)

parser.add_argument('--dt', default=1.0, type=float)
parser.add_argument('--simulationT', default=600000, type=int)

parser.add_argument('--num_Xe', default=10, type=int)
parser.add_argument('--num_Xi', default=10, type=int)
parser.add_argument('--num_Se', default=80, type=int)
parser.add_argument('--num_Si', default=20, type=int)

args = parser.parse_args()
duration=100; bin_size=1

# path
#=================================================================================================================
root_path = os.getcwd()
data_path = os.path.join(root_path, args.dataFolderName)
segments_path = os.path.join(data_path, args.segFolderName)
ccgResult_path = os.path.join(segments_path, 'CCG'); os.makedirs(ccgResult_path, exist_ok=True)
ccgPlot_path = os.path.join(segments_path, 'CCG_Plots'); os.makedirs(ccgPlot_path, exist_ok=True)


# arguments
#=================================================================================================================
T = args.simulationT
dt = args.dt
Lt = int(T/dt)
range_t = np.arange(0, T, dt)

cmap= 'coolwarm'; cm_cmap='PuBu'; 
mC='firebrick'; bC='steelblue'; ccgColor='k'
coolwarm_blue = '#3B4CC0'; coolwarm_red = '#B40426'
coolwarm_softblue = '#6BAED6'; coolwarm_softred = '#E07A7A'; correctGREEN= '#2E7D32'

# load
#=================================================================================================================
Xe, Xi = args.num_Xe, args.num_Xi 
Se, Si = args.num_Se, args.num_Si 
N = Xe + Se + Xi+Si
effectiveN = Se + Si
effectiveName_list = ['SE'+str(i) for i in range(Se)] + ['SI'+str(i) for i in range(Si)]

hdf5_name = 'T'+str(T)+'_N'+str(N)+'.h5'
with pd.HDFStore(os.path.join(segments_path, hdf5_name) , mode="r") as store:
    
    spkTimes_df = store["spkTimes_df"]
    CM_clean = store["CM_clean"]

CM_clean = CM_clean.to_numpy()
featureSummary = pd.read_excel(os.path.join(segments_path, 'featureSummary.xlsx')).iloc[:, 1:]
print('Done Loading DFs. CM_clean.shape:', CM_clean.shape, 'spkTimes_df.shape:', spkTimes_df.shape)


# CCG Calculation
#=================================================================================================================
saveBin=True
cch_dic={}; cchDiff_dic={}; ach1_dic={}; ach2_dic={}; ahc1_nspk={}; ahc2_nspk={}; GLMCC_graphs=[]; GLMPP_graphs=[]
startTime = time.time()
for i in range(effectiveN):
    name_i = effectiveName_list[i]
    spkTimes_i = spkTimes_df[name_i].dropna()
    neuronType_i = 'Exc' if 'SE' in name_i else "Inh"
    fr_i, cv_i = featureSummary[featureSummary['neuron_name']==name_i][['firing_rate', 'cv_ISI']].values[0]

    # this neuron project to
    cs_i = np.where(CM_clean[i]!=0)[0]
    # this neuron doesn't projec to (same len with cs_i)
    ncs_i = np.where(CM_clean[i]==0)[0]
    ncs_i = ncs_i[np.random.randint(0, len(ncs_i), len(cs_i))]
    # use these samples to generate ccg
    ccg_i_list = np.concatenate((cs_i, ncs_i))
    
    for j in ccg_i_list:
        name_j = effectiveName_list[j]
        spkTimes_j = spkTimes_df[name_j].dropna()
        neuronType_j = 'Exc' if 'SE' in name_j else "Inh"
        fr_j, cv_j = featureSummary[featureSummary['neuron_name']==name_j][['firing_rate', 'cv_ISI']].values[0]

        if fr_i>0.5 and fr_j>0.5:
            w_i2j = CM_clean[i,j]
            title = name_i+':fr='+str(round(fr_i, 3))+',CV='+str(round(cv_i, 3)) + '---W='+str(w_i2j)+'---'+name_j+':fr='+str(round(fr_j, 3))+',CV='+str(round(cv_j, 3)) if w_i2j!=0 else name_i+'-'+name_j+'-NoConnection'
            saveName = f'{name_i}-{name_j}-c' if w_i2j!=0 else f'{name_i}-{name_j}-n'

            times, idx, nspks1, nspks2 = merge_spkTimes(spkTimes_i, spkTimes_j)
            
            # CCG & Plot------------------------
            ccg_result = cal_CCG(times, idx, duration=duration, bin_size=bin_size, ifPlot=False, ifSave=False)
            cch, cch_diffs, ach1, ach2, n_bins, half_bins, t = ccg_result
            cch_dic[name_i+'_'+name_j] = np.concatenate((cch, [w_i2j]))
            
            plot_ccg(ccg=cch, timeBins=t, title=title, barColor=ccgColor, ifSave=True, savePath=ccgPlot_path, filename=saveName)



            if saveBin:
                info = {'n_bins':n_bins, 'half_bins':half_bins, 't':t }
                info_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in info.items() ]))
                info_df.to_csv(os.path.join(ccgResult_path, 'binInfo.csv'))
                saveBin=False


CCH_df = pd.DataFrame.from_dict(cch_dic, orient='index').T; CCH_df.to_csv(os.path.join(ccgResult_path, 'CCH.csv'))


endTime = time.time()
print('Done CCG. Cost', round((endTime-startTime)/60, 3))



