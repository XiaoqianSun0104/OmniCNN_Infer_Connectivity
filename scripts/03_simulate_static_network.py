"""
# 03_simulate_static_network.py
# Author: Xiaoqian Sun, 01/27/2025
# generate network simulation
# Pipeline stage script for OmniCNN connectivity inference -- python scripts/02_simulate_static_network.py --help
"""


# Import Packages
#=================================================================================================================
import os
import time
import argparse
import numpy as np
import pandas as pd
from Functions import *
import matplotlib.pyplot as plt

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "PointNeuron_Simulation"))

import signals, neuron, connectivity, simulation
from utils import *; from visualization import *; from connectivity import *



import warnings
warnings.filterwarnings('ignore')


#===================================================================================================#
#                                            Prepare                                                #
#===================================================================================================#
parser = argparse.ArgumentParser("Static Network Simulation")
parser.add_argument('--dataFolderName', default='data', type=str)
parser.add_argument('--segFolderName', default='static_network', type=str)

parser.add_argument('--dt', default=1.0, type=float)
parser.add_argument('--simulationT', default=600000, type=int)
parser.add_argument('--num_Xe', default=10, type=int)
parser.add_argument('--num_Xi', default=10, type=int)
parser.add_argument('--num_Se', default=80, type=int)
parser.add_argument('--num_Si', default=20, type=int)

parser.add_argument('--tau_m_exc', default=15., type=float)
parser.add_argument('--gLeak_exc', default=10., type=float)
parser.add_argument('--tau_excSyn', default=3., type=float)

parser.add_argument('--tau_m_inh', default=8., type=float)
parser.add_argument('--gLeak_inh', default=15., type=float)
parser.add_argument('--tau_inhSyn', default=5.7, type=float)

parser.add_argument('--p_exc2exc', default=0.1, type=float)
parser.add_argument('--p_exc2inh', default=0.25, type=float)
parser.add_argument('--p_inh2exc', default=0.25, type=float)
parser.add_argument('--p_inh2inh', default=0.1, type=float)

parser.add_argument('--gE2E_bar_scaler', default=1.0, type=float)
parser.add_argument('--gE2I_bar_scaler', default=3.5, type=float)

parser.add_argument('--gI2E_bar_scaler', default=2.0, type=float)
parser.add_argument('--gI2I_bar_scaler', default=1.5, type=float)

parser.add_argument('--mu_Se', default=180., type=float)
parser.add_argument('--mu_Si', default=200., type=float)

parser.add_argument('--plotCCGs', action='store_true', help='include --plotCCGs in script, code will generate CCG visualizations')


args = parser.parse_args()
burstThreshold = 7
duration=100; bin_size=1
cv_no=0.2; cv_low=0.7; cv_poisson=1; bf_mid=0.4; bf_high=0.7
cmap= 'coolwarm'; cm_cmap='PuBu'; mC='firebrick'; bC='steelblue'; ccgColor='lightslategrey'

# path
#====================================================================================================
root_path = str(PROJECT_ROOT)
data_path = os.path.join(root_path, args.dataFolderName); os.makedirs(data_path, exist_ok=True)
segments_path = os.path.join(data_path, args.segFolderName); os.makedirs(segments_path, exist_ok=True)
ccgResult_path = os.path.join(segments_path, 'CCG'); os.makedirs(ccgResult_path, exist_ok=True)
ccgPlot_path = os.path.join(segments_path, 'CCG_Plots'); os.makedirs(ccgPlot_path, exist_ok=True)



#===================================================================================================#
#                                            Arguments                                              #
#===================================================================================================#

# Whole Simulation
# --------------------------------------------------------------------------------------
T = args.simulationT
dt = args.dt
Lt = int(T/dt)
range_t = np.arange(0, T, dt)
maxns = int(1e6)

Xe, Xi = args.num_Xe, args.num_Xi 
Se, Si = args.num_Se, args.num_Si 
Ne, Ni = Xe + Se, Xi+Si
N = Ne + Ni
excNeuronType, inhNeuronType = 0, 1

# NeuronObj Info
neuronType_list = [0]*Ne + [1]*Ni
Xe_neuronNames = ['XE'+str(i) for i in range(Xe)]; Xi_neuronNames = ['XI'+str(i) for i in range(Xi)]
Se_neuronNames = ['SE'+str(i) for i in range(Se)]; Si_neuronNames = ['SI'+str(i) for i in range(Si)]
neuronName_list = Xe_neuronNames + Se_neuronNames + Xi_neuronNames + Si_neuronNames

cm_kwargs = {'cols':neuronName_list,
             'w_exc2exc':1., 'w_exc2inh':1., 'w_inh2exc':1., 'w_inh2inh':1., 
             'we2e_max':5., 'we2i_max':3., 'wi2e_max':8., 'wi2i_max':2.,
             'p_exc2exc':args.p_exc2exc, 'p_exc2inh':args.p_exc2inh, 'p_inh2exc':args.p_inh2exc, 'p_inh2inh':args.p_inh2inh
            }
pXe2e = 0.12; pXe2i = 0.1; pXi2e = 0.1; pXi2i=0.1

w_Xe2e_lb = 1.0; w_Xe2i_lb = 1.0; w_Xi2e_lb = 0.5; w_Xi2i_lb = 0.5
w_Xe2e_ub = 1.5; w_Xe2i_ub = 1.0; w_Xi2e_ub = 1.0; w_Xi2i_ub = 1.0

w_Se2e_lb = 1.0; w_Se2i_lb = 1.5; w_Si2e_lb = 1.0; w_Si2i_lb = 0.5
w_Se2e_ub = 2.0; w_Se2i_ub = 2.5; w_Si2e_ub = 1.5; w_Si2i_ub = 1.0

ce2e = int(cm_kwargs['p_exc2exc']*Se); ce2i = int(cm_kwargs['p_exc2inh']*Si); ci2e = int(cm_kwargs['p_inh2exc']*Se); ci2i = int(cm_kwargs['p_inh2inh']*Si)
cXe2e = int(pXe2e*Se); cXe2i = int(pXe2i*Si); cXi2e = int(pXi2e*Se); cXi2i = int(pXi2i*Si)


# conductance level (inhbitory conductance is typically stronger)
scalingFactor = np.sqrt((Se+Si))
g_bar = 2.5/scalingFactor # nS
gE2E_bar, gE2I_bar = args.gE2E_bar_scaler*g_bar, args.gE2I_bar_scaler*g_bar 
gI2E_bar, gI2I_bar = args.gI2E_bar_scaler*g_bar, args.gI2I_bar_scaler*g_bar

# others
burstThreshold = 7
cv_no=0.2; cv_low=0.7; cv_poisson=1; bf_mid=0.4; bf_high=0.7



# ==========================================XE/XI==========================================#
Seeds_Xe = 1992+np.array(range(Xe)); Seeds_Xi = 1886+np.array(range(Xi))
FRs_Xe = np.random.uniform(20, 30, Xe); FRs_Xi = np.random.uniform(25, 30, Xi)
BPs_Xe = np.random.uniform(0.8, 1.0, Xe); BPs_Xi = np.random.uniform(0.6, 0.8, Xe)


# ===========================================SE/SI=========================================#
tau_m_exc = args.tau_m_exc; gLeak_exc = args.gLeak_exc; tau_excSyn = args.tau_excSyn
tau_m_inh = args.tau_m_inh;  gLeak_inh = args.gLeak_inh; tau_inhSyn = args.tau_inhSyn
VE_inits = sample_gaussian(Se, -65, 2, -70, -60)
VI_inits = sample_gaussian(Si, -65, 2, -70, -60)

# exteral noisy input 
I_b_Se = []; I_b_Si = []
mu_Se=180; mu_Si=200
tauNoises_Se = np.random.uniform(15, 30, Se); tauNoises_Si = np.random.uniform(25, 30, Si)
sigmaNoises_Se = np.random.uniform(3, 5, Se); sigmaNoises_Si = np.random.uniform(3, 5, Si)
for i in range(Se):
    I_b_Se.append(signals.noisyOU_input(Lt, dt, mu_noise=mu_Se, 
                                      tau_noise=tauNoises_Se[i], sigma_noise=sigmaNoises_Se[i] ))
for i in range(Si):
    I_b_Si.append(signals.noisyOU_input(Lt, dt, mu_noise=mu_Si, 
                                      tau_noise=tauNoises_Si[i], sigma_noise=sigmaNoises_Si[i] ))


#===================================================================================================#
#                                            Simulation                                             #
#===================================================================================================#
startTime = time.time()

# ==========================================XE/XI===================================================#
Xe_kwargs = [{'lr_stdp_postSyn':0, 'lr_stdp_preSyn':0,'lr_istdp':0,
              'ifAssignSpkTrain':True, 'rate':FRs_Xe[i], 'poissonSpk_seed':Seeds_Xe[i], 
              'ifARP':True, 'ifBursting':True, 'BP':BPs_Xe[i]} for i in range(Xe)]

Xi_kwargs = [{'lr_stdp_postSyn':0, 'lr_stdp_preSyn':0,'lr_istdp':0,
              'ifAssignSpkTrain':True, 'rate':FRs_Xi[i], 'poissonSpk_seed':Seeds_Xi[i], 
              'ifARP':True, 'ifBursting':True, 'BP':BPs_Xi[i]} for i in range(Xi)]

Xe_Objs = [neuron.Neuron(T, dt, g_bar, g_bar, excNeuronType, **xe_kwarg) for xe_kwarg in Xe_kwargs]
Xi_Objs = [neuron.Neuron(T, dt, g_bar, g_bar, inhNeuronType, **xi_kwarg) for xi_kwarg in Xi_kwargs]


# ===========================================SE/SI===================================================#
Se_kwargs = [{'tau_m':tau_m_exc, 'tref':2., 'g_Leak':gLeak_exc, 'tau_excSyn':tau_excSyn, 'tau_inhSyn':tau_inhSyn, 
              'lr_stdp_postSyn':0, 'lr_stdp_preSyn':0, 'lr_istdp':0, 'lr_istdp_window':0,
              'V_init': VE_inits[i],'externalInput':I_b_Se[i]} for i in range(Se)]
Se_Objs = []
for i in range(Se):
    se_kwarg = Se_kwargs[i]; Se_Objs.append(neuron.Neuron(T, dt, gE2E_bar, gI2E_bar, excNeuronType,  **se_kwarg))


Si_kwargs = [{'tau_m':tau_m_inh, 'tref':2., 'g_Leak':gLeak_inh, 'tau_excSyn':tau_excSyn, 'tau_inhSyn':tau_inhSyn, 
              'lr_stdp_postSyn':0, 'lr_stdp_preSyn':0, 'lr_istdp':0, 'lr_istdp_window':0,
              'V_init': VI_inits[i],'externalInput':I_b_Si[i]} for i in range(Si)]
Si_Objs = []
for i in range(Si):
    si_kwarg = Si_kwargs[i]; Si_Objs.append(neuron.Neuron(T, dt, gE2I_bar, gI2I_bar, inhNeuronType,  **si_kwarg))


S_Obj_list = Se_Objs + Si_Objs
neuronObj_list = Xe_Objs + Se_Objs + Xi_Objs + Si_Objs
print('--', len(neuronObj_list), 'neurons in the network')


# ==================================Connectivity Matrix==============================================#
connectivityM = np.zeros((N, N))
for i in range(0, Xe): # for each Xe neuron
    connectivityM[i, generate_randomInts(Xe, Xe+Se-1, cXe2e, i)] = np.random.uniform(w_Xe2e_lb, w_Xe2e_ub, cXe2e)
    connectivityM[i, generate_randomInts(Xe+Se+Xi, Xe+Se+Xi+Si-1, cXe2i, i)] = np.random.uniform(w_Xe2i_lb, w_Xe2i_ub, cXe2i)
for i in range(Xe, Xe+Se): # for each Se neuron
    connectivityM[i, generate_randomInts(Xe, Xe+Se-1, ce2e, i)] = np.random.uniform(w_Se2e_lb, w_Se2e_ub, ce2e)
    connectivityM[i, generate_randomInts(Xe+Se+Xi, Xe+Se+Xi+Si-1, ce2i, i)] = np.random.uniform(w_Se2i_lb, w_Se2i_ub, ce2i)

for i in range(Xe+Se, Xe+Se+Xi): # for each Xi neuron
    connectivityM[i, generate_randomInts(Xe, Xe+Se-1, cXi2e, i)] = np.random.uniform(w_Xi2e_lb, w_Xi2e_ub, cXi2e)
    connectivityM[i, generate_randomInts(Xe+Se+Xi, Xe+Se+Xi+Si-1, cXi2i, i)] = np.random.uniform(w_Xi2i_lb, w_Xi2i_ub, cXi2i)
for i in range(Xe+Se+Xi, Xe+Se+Xi+Si): # for each Si neuron
    connectivityM[i, generate_randomInts(Xe, Xe+Se-1, ci2e, i)] = np.random.uniform(w_Si2e_lb, w_Si2e_ub, ci2e)
    connectivityM[i, generate_randomInts(Xe+Se+Xi, Xe+Se+Xi+Si-1, ci2i, i)] = np.random.uniform(w_Si2i_lb, w_Si2i_ub, ci2i)



# ====================================simulator======================================================#
simulator = simulation.Simulator(T, dt, N, Ne, Ni, maxns, neuronObj_list, CM=connectivityM, **cm_kwargs)
simuResult = simulator.run(ifVerbose=False, pickN=None)
netSpk = simulator.get('netSpk')
netSpk_Summary = generate_netSpk_report(T, netSpk, neuronName_list, neuronType_list, ifSave=False, savePath=None, filename='')

endTime = time.time()
print('--Simulate T='+str(T)+' cost', round((endTime-startTime)/60, 3))


# ====================================Plot & get DFs==================================================#
CM = simulator.get('CM')
CM_clean = clip_externalNeuron(CM, Xe, Xi, Se, Si)

# multiple by g_bar (real synaptic weight)
CM_clean[0:Se, 0:Se] = CM_clean[0:Se, 0:Se]*gE2E_bar
CM_clean[0:Se, Se:Se+Si] = CM_clean[0:Se, Se:Se+Si]*gE2I_bar
CM_clean[Se:Se+Si, 0:Se] = CM_clean[Se:Se+Si, 0:Se]*gI2E_bar*(-1)
CM_clean[Se:Se+Si, Se:Se+Si] = CM_clean[Se:Se+Si, Se:Se+Si]*gI2I_bar*(-1)

CM_clean_df = pd.DataFrame(CM_clean, columns=Se_neuronNames + Si_neuronNames, index = Se_neuronNames + Si_neuronNames)
CM_clean_df.to_excel(os.path.join(segments_path, 'CMW_SeSi.xlsx'))
plot_CM(CM_clean, Se, Si, eColor='magenta', iColor='cyan', figsize=(10, 10), ifSave=True, savePath=segments_path, filename='CMW')


## Get Simulated (SE/SI) Neurons'DFs
S_dfs = []
for key in ['spkTrain', 'spkTimes', 'memPotential', 'EPSP', 'IPSP', 'Xcurr', 'LeakC', 'synTrace', 'gE', 'gI']:
    array = get_keyValues_2_2DArray(S_Obj_list, list(range(Se+Si)), T, dt, Key=key) 
    
    df = pd.DataFrame(array).T
    df.columns = Se_neuronNames + Si_neuronNames
    if key=='spkTimes': df = df.loc[~(df==0).all(axis=1)]

    print('--', key+'.shape:', df.shape, end=' | ')
    S_dfs.append(df)

spkTrain_df=S_dfs[0]; spkTimes_df=S_dfs[1]; memPotential_df=S_dfs[2]
EPSP=S_dfs[3]; IPSP=S_dfs[4]; Xcurr=S_dfs[5]; LeakC=S_dfs[6]; 
synTrace = S_dfs[7]; gE = S_dfs[8]; gI = S_dfs[9]


#===================================================================================================#
#                                        Feature Summary                                            #
#===================================================================================================#
# Evaluation of Simulated Neuron
cols=['V_init', 'gE_bar', 'gI_bar' ]
paras = extract_neuronParas_fromObjs(S_Obj_list, keys=cols, neuronNames=Se_neuronNames+Si_neuronNames, ifSave=False)
paras = paras.apply(pd.to_numeric, errors="ignore"); paras = paras.reset_index()

incomingCs = simulator.get('incomingCs')
incomingNs_dic = {}; numXes = []; numXis = []; numSes = []; numSis = []
for i in list(range(Xe, Xe+Se))+list(range(Xe+Se+Xi, Xe+Se+Xi+Si)):
    neuronName = neuronName_list[i]
    incmoingcs_2_i = incomingCs[i]
    incomingNs_2_i = [neuronName_list[k] for k in incmoingcs_2_i]

    count_Xe = sum(1 for item in incomingNs_2_i if 'XE' in item); numXes.append(count_Xe)
    count_Xi = sum(1 for item in incomingNs_2_i if 'XI' in item); numXis.append(count_Xi)
    count_Se = sum(1 for item in incomingNs_2_i if 'SE' in item); numSes.append(count_Se)
    count_Si = sum(1 for item in incomingNs_2_i if 'SI' in item); numSis.append(count_Si)

    incomingNs_dic[neuronName]  = incomingNs_2_i

incomingNs_DF = pd.DataFrame(list(incomingNs_dic.items()), columns=['neuron_name', 'incomingCs'])
incomingNs_DF['numXe'] = numXes; incomingNs_DF['numXi'] = numXis; incomingNs_DF['numSe'] = numSes; incomingNs_DF['numSi'] = numSis

ISI_list = []
for i in range(spkTimes_df.shape[1]):
    st = spkTimes_df.iloc[:,i].values
    _, _, burstingFraction, mean_ISI, _, _, cv_ISI, _= interSpikeIntervals_Stats(st,burstThreshold=burstThreshold, ifVerbose=False, ifPlotHist=False)
    feature_dic = {'burstingFraction':burstingFraction, 'mean_ISI': mean_ISI,  'cv_ISI': cv_ISI, }
    ISI_list.append(pd.DataFrame.from_dict(feature_dic, orient='index'))

ISIs = pd.concat(ISI_list, axis=1).T.reset_index(drop=True)
ISIs['neuron_name'] = paras['neuron_name']

attencols = [ 'neuron_type', 'neuron_name', 'V_init', 'gE_bar', 'gI_bar', 'firing_rate', 'cv_ISI', 'burstingFraction', 'firing_times']
feature_summary = pd.merge(paras, ISIs, on="neuron_name", how="outer")
feature_summary = pd.merge(feature_summary, netSpk_Summary, on="neuron_name", how="left")
feature_summary = feature_summary[attencols]; feature_summary = pd.merge(feature_summary, incomingNs_DF, on="neuron_name", how="left");
feature_summary.to_excel(os.path.join(segments_path, 'feature_summary.xlsx'))

# Histogram
SE_CVs = feature_summary[feature_summary['neuron_name'].str.contains('SE')]['cv_ISI']
SE_FRs = feature_summary[feature_summary['neuron_name'].str.contains('SE')]['firing_rate']
SI_CVs = feature_summary[feature_summary['neuron_name'].str.contains('SI')]['cv_ISI']
SI_FRs = feature_summary[feature_summary['neuron_name'].str.contains('SI')]['firing_rate']
data_list = [feature_summary['firing_rate'],feature_summary['cv_ISI'],SE_FRs,SE_CVs, SI_FRs, SI_CVs] 
label_list = ['All-FR', 'All-CV', 'Exc-FR', 'Exc-CV', 'Inh-FR', 'Inh-CV']; color_list = ['g', 'g', 'orange', 'orange', 'b', 'b']
figR=3; figC=2; figsize=(8, 6)
subplots_histos(data_list, label_list, color_list, figR=figR, figC=figC, figSize=figsize, bins=30, 
                ifSave=True, savePath=segments_path, filename='Histos.png')


# E-I Balance Analysis (global)
EI_ratio = cal_global_EI_ratio(gE.filter(like='SE', axis=1), gI.filter(like='SE', axis=1), T, dt, 
                               ifPlot=True, ifSave=True, savePath=segments_path, filename='global_gEgI')
lags, cross_corr = cal_XCorr_EI_Current(EPSP.filter(like='SE', axis=1), IPSP.filter(like='SE', axis=1), T, dt, 
                                        ifPlot=True, ifSave=True, savePath=segments_path, filename='XCorr_EICurrent')

# E-I Balance Analysis (single neuron)
neuronNames = ['SE'+str(i) for i in np.random.randint(1, Se-1, size=5)]
fig, ax = plt.subplots(len(neuronNames), 1, figsize=(20, 3*len(neuronNames)))
for i in range(len(neuronNames)):
    neuronName = neuronNames[i]
    ax[i].plot(gE[neuronName][3000:5000], c='b', label='gE')
    ax[i].plot(-1*gI[neuronName][3000:5000], c='orange', label='gI')
    ax[i].set_title(neuronName, fontweight='bold'); ax[i].legend(loc=1)
plt.tight_layout(); plt.savefig(os.path.join(segments_path, 'neuron_gEgI')); plt.close()

fig, ax = plt.subplots(len(neuronNames), 1, figsize=(20, 3*len(neuronNames)))
for i in range(len(neuronNames)):
    neuronName = neuronNames[i]
    ax[i].plot(EPSP[neuronName][3000:5000], c='b', label='ECurr')
    ax[i].plot(IPSP[neuronName][3000:5000], c='orange', label='ICurr')
    ax[i].set_title(neuronName, fontweight='bold'); ax[i].legend(loc=1)
plt.tight_layout(); plt.savefig(os.path.join(segments_path, 'neuron_ECIC')); plt.close()

# global E/Icurr distribution
plt.figure(figsize=(6, 4))
plt.hist(EPSP.filter(like='SE', axis=1).mean(axis=1), facecolor='k', alpha=0.8, label='mean Ecurr')
plt.hist(IPSP.filter(like='SE', axis=1).mean(axis=1), facecolor='orange', alpha=0.8, label='mean Icurr')
plt.legend(); plt.savefig(os.path.join(segments_path, 'ECIC_Hist')); plt.close()

print('--Done Generating Feature.')


#===================================================================================================#
#                                              Saving                                               #
#===================================================================================================#
hdf5_name = 'T'+str(T)+'_N'+str(N)+'.h5'
with pd.HDFStore(os.path.join(segments_path, hdf5_name) , mode="w") as store:

    store.put("spkTimes_df", spkTimes_df, format="table")
    store.put("memPotential_df", memPotential_df, format="table",  complevel=9, complib="blosc")

    store.put("EPSP", EPSP, format="table")
    store.put("IPSP", IPSP, format="table")
    store.put("Xcurr", Xcurr, format="table")
    store.put("LeakC", LeakC, format="table")

    store.put("gE", gE, format="table")
    store.put("gI", gI, format="table")
    store.put("synTrace", synTrace, format="table")
    
    store.put("CM_df", simulator.get('CM_df'), format="table")
    store.put("CM_clean", pd.DataFrame(CM_clean), format="table")    

print('--Done Saving Results.')



#===================================================================================================#
#                                         CCG Calculation                                           #
#===================================================================================================#
effectiveN = Se + Si
effectiveName_list = Se_neuronNames + Si_neuronNames
saveBin=True; cch_dic={}; startTime = time.time()

for i in range(effectiveN):
    name_i = effectiveName_list[i]
    spkTimes_i = spkTimes_df[name_i].dropna()
    neuronType_i = 'Exc' if 'SE' in name_i else "Inh"
    fr_i, cv_i = feature_summary[feature_summary['neuron_name']==name_i][['firing_rate', 'cv_ISI']].values[0]

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
        fr_j, cv_j = feature_summary[feature_summary['neuron_name']==name_j][['firing_rate', 'cv_ISI']].values[0]

        if fr_i>0.5 and fr_j>0.5:
            w_i2j = CM_clean[i,j]
            title = name_i+':fr='+str(round(fr_i, 3))+',CV='+str(round(cv_i, 3)) + '---W='+str(w_i2j)+'---'+name_j+':fr='+str(round(fr_j, 3))+',CV='+str(round(cv_j, 3)) if w_i2j!=0 else name_i+'-'+name_j+'-NoConnection'
            saveName = f'{name_i}-{name_j}-c' if w_i2j!=0 else f'{name_i}-{name_j}-n'

            times, idx, nspks1, nspks2 = merge_spkTimes(spkTimes_i, spkTimes_j)
            
            # CCG & Plot------------------------
            ccg_result = cal_CCG(times, idx, duration=duration, bin_size=bin_size, ifPlot=False, ifSave=False)
            cch, cch_diffs, ach1, ach2, n_bins, half_bins, t = ccg_result
            cch_dic[name_i+'_'+name_j] = np.concatenate((cch, [w_i2j]))
            if args.plotCCGs:
                plot_ccg(ccg=cch, timeBins=t, title=title, barColor=ccgColor, ifSave=True, savePath=ccgPlot_path, filename=saveName)

            if saveBin:
                info = {'n_bins':n_bins, 'half_bins':half_bins, 't':t }
                info_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in info.items() ]))
                info_df.to_csv(os.path.join(ccgResult_path, 'binInfo.csv'))
                saveBin=False

CCH_df = pd.DataFrame.from_dict(cch_dic, orient='index').T; CCH_df.to_csv(os.path.join(ccgResult_path, 'CCH.csv'))

endTime = time.time()

print(f'--Done CCG. Cost {round((endTime-startTime)/60, 3)} \n')

