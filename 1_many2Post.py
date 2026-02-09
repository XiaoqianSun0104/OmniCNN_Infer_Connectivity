# 1_many2Post.py
# Author: Xiaoqian Sun, 03/25/2025
# Fucntion: simulation of 40->1 small motif 


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
from scipy.stats import skew, kurtosis, entropy


# simulation package
import sys 
import importlib
import utils, signals, connectivity, neuron, visualization, simulation
importlib.reload(utils)
importlib.reload(neuron)
importlib.reload(signals)
importlib.reload(connectivity)
importlib.reload(visualization)
importlib.reload(simulation_addOn)
from utils import *; from visualization import *

import warnings
warnings.filterwarnings('ignore')


startTime = time.time()

# argumetns
#=================================================================================================================
parser = argparse.ArgumentParser("Replicates Generation")
parser.add_argument('--dataFolderName', default='C_Data', type=str)
parser.add_argument('--segFolderName', default='m2post', type=str)

parser.add_argument('--seedStart', default=1986, type=int)

parser.add_argument('--dt', default=1.0, type=float)
parser.add_argument('--simulationT', default=600000, type=int)
parser.add_argument('--num_pre', default=40, type=int)
parser.add_argument('--per_preE', default=0.5, type=float)
parser.add_argument('--per_preI', default=0.5, type=float)
parser.add_argument('--num_post', default=1, type=int)

parser.add_argument('--sampleMethod', default='linspace', type=str)
parser.add_argument('--pre_FR', default=20., type=float)
parser.add_argument('--preBP_lb', default=0.8, type=float)
parser.add_argument('--preBP_ub', default=1.0, type=float)


parser.add_argument('--gE2E_bar', default=1.0, type=float)
parser.add_argument('--gI2E_bar', default=0.7, type=float)
parser.add_argument('--wij_lb', default=0.2, type=float)
parser.add_argument('--wij_ub', default=2., type=float)

parser.add_argument('--mu_noise', default=150, type=float)
parser.add_argument('--tau_noise', default=15, type=float)
parser.add_argument('--sigma_noise', default=5, type=float)

parser.add_argument('--burstThreshold', default=7., type=float)

parser.add_argument('--generateOtherCCGs', action='store_true', help='include --generateOtherCCGs in script, code will generate all kinds of CCGs')
parser.add_argument('--generateUnconnectedCCG', action='store_true', help='include --generateUnconnectedCCG in script, code will generate antoher 40 pairs of CCG among presynaptic neruons')


args = parser.parse_args()

burstThreshold = args.burstThreshold
cv_no=0.2; cv_low=0.7; cv_poisson=1; bf_mid=0.4; bf_high=0.7
duration=100; bin_size=1
cmap= 'coolwarm'; cm_cmap='PuBu'; mC='firebrick'; bC='steelblue'; ccgColor='lightslategrey'

# path
#=================================================================================================================
root_path = os.getcwd()
data_path = os.path.join(root_path, args.dataFolderName)
segments_path = os.path.join(data_path, args.segFolderName)
ccgResult_path = os.path.join(segments_path, 'CCG')

if not os.path.exists(data_path): os.makedirs(data_path)
if not os.path.exists(segments_path): os.makedirs(segments_path)
if not os.path.exists(ccgResult_path): os.makedirs(ccgResult_path)

#=================================================================================================================
#                                                   Arguments                                                    #
#=================================================================================================================
# Whole Simulation--------------------------------------
T = args.simulationT      
dt = args.dt
Lt = int(T/dt)
maxns=int(1e6)
range_t = np.arange(0, T, dt)
excNeuronCode=0; inhNeuronCode=1

num_pre = args.num_pre
per_preE = args.per_preE; num_preE = int(per_preE * num_pre)
per_preI = args.per_preI; num_preI = int(per_preI * num_pre)

num_post = args.num_post
Ne=num_preE+num_post; Ni=num_preI; N=Ne+Ni

neuronType_list = [excNeuronCode]*num_preE + [excNeuronCode]*num_post + [inhNeuronCode]*num_preI
neuronName_preE = ['Pre_E'+str(i) for i in range(num_preE)]
neuronName_preI = ['Pre_I'+str(i) for i in range(num_preI)]
neuronName_pre = neuronName_preE + neuronName_preI; neuronName_post = ['Post']
# make sure 1st half is E (pre-assigned + simulated), 2nd half is I
# because the connectivity assumed that first half is E, 2nd half is I
neuronName_list = neuronName_preE + neuronName_post + neuronName_preI 

# Presynaptic-------------------------------------------
FR_E = args.pre_FR; FR_I = args.pre_FR+5; seedStart = args.seedStart
SeedsE = seedStart+np.array(range(num_preE));SeedsI = seedStart+1979+np.array(range(num_preI))

if args.sampleMethod == 'uniform':
    BPs_E = np.random.uniform(args.preBP_lb, args.preBP_ub, num_preE)
    BPs_I = np.random.uniform(args.preBP_lb, args.preBP_ub, num_preI)
elif args.sampleMethod == 'linspace':
    BPs_E = np.linspace(args.preBP_lb, args.preBP_ub, num_preE); np.random.shuffle(BPs_E)
    BPs_I = np.linspace(args.preBP_lb, args.preBP_ub, num_preI); np.random.shuffle(BPs_I)

# Postsynaptic------------------------------------------
gE2E_bar = args.gE2E_bar; gI2E_bar = args.gI2E_bar
mu_noise=args.mu_noise; tau_noise = args.tau_noise; sigma_noise = args.sigma_noise
I_b = signals.noisyOU_input(Lt, dt, mu_noise=mu_noise, tau_noise=tau_noise, sigma_noise=sigma_noise )

if args.sampleMethod == 'uniform':
    w_E = np.random.uniform(args.wij_lb, args.wij_ub, num_preE)
    w_I = np.random.uniform(args.wij_lb, args.wij_ub, num_preI)
elif args.sampleMethod == 'linspace':
    w_E = np.linspace(args.wij_lb, args.wij_ub, num_preE); np.random.shuffle(w_E)
    w_I = np.linspace(args.wij_lb, args.wij_ub, num_preI); np.random.shuffle(w_I)
gE2Es = gE2E_bar * w_E; gI2Es = gI2E_bar * w_I; g_s = np.concatenate([gE2Es, -1*gI2Es, np.array([np.nan])])



# =================================================================================================================
#                                                  Simulation                                                     #
# =================================================================================================================
# Presynaptic --------------------------------------------------------------------------------------
preE_kwargs = [{'lr_stdp_postSyn':0, 'lr_stdp_preSyn':0,'lr_istdp':0, 
               'ifAssignSpkTrain':True, 'rate':FR_E, 'poissonSpk_seed':SeedsE[i], 
               'ifARP':True, 'ifBursting':True, 'BP':BPs_E[i]} for i in range(num_preE)]
preE_Objs = [neuron.Neuron(T, dt, gE2E_bar, gI2E_bar, 0, **pre_kwarg) for pre_kwarg in preE_kwargs]

preI_kwargs = [{'lr_stdp_postSyn':0, 'lr_stdp_preSyn':0,'lr_istdp':0, 
               'ifAssignSpkTrain':True, 'rate':FR_I, 'poissonSpk_seed':SeedsI[i], 
               'ifARP':True, 'ifBursting':True, 'BP':BPs_I[i]} for i in range(num_preI)]
preI_Objs = [neuron.Neuron(T, dt, gE2E_bar, gI2E_bar, 1, **pre_kwarg) for pre_kwarg in preI_kwargs]

pre_Objs = preE_Objs + preI_Objs

# spkTrain / spkTimes ---------------------------------
spkTrain_pre = get_keyValues_2_2DArray(pre_Objs, list(range(num_pre)), T, dt, Key='assignSpkTrain') 
spkTrainPre_df = pd.DataFrame(spkTrain_pre).T; spkTrainPre_df.columns = neuronName_preE + neuronName_preI

spkTimes_list = []
for i in range(spkTrainPre_df.shape[1]):
    t_sp = np.where(spkTrainPre_df.iloc[:,i] > 0.5)[0]
    spkTimes_list.append(t_sp*dt)
max_len = max(len(arr) for arr in spkTimes_list)
spkTimes_list = [np.pad(arr, (0, max_len - len(arr)), constant_values=0) for arr in spkTimes_list]
spkTimesPre_df = pd.DataFrame(spkTimes_list).T; spkTimesPre_df.columns = spkTrainPre_df.columns



# Postsynaptic --------------------------------------------------------------------------------------
post_kwargs = {'tau_m':15, 'lr_stdp_postSyn':0, 'lr_stdp_preSyn':0,'lr_istdp':0, 'externalInput':I_b}
post_Obj = [ neuron.Neuron(T, dt, gE2E_bar, gI2E_bar, 0,  **post_kwargs) ]
# make sure first half is E (assigned or simulated), 2nd half is I
neuronObj_list = preE_Objs + post_Obj + preI_Objs 

# Connectivity Matrix-----------------------------------
connectivityM = np.zeros((N, N))
connectivityM[0:num_preE, num_preE:Ne] = w_E.reshape(-1, 1)
connectivityM[Ne:N, num_preE:Ne] = w_I.reshape(-1, 1)
cm_kwargs = {'w_exc2exc':1, 'w_exc2inh':1, 'w_inh2exc':1, 'w_inh2inh':1}

# Simulator---------------------------------------------
simulator = simulation.Simulator(T, dt, N, Ne, Ni, maxns, neuronObj_list, CM=connectivityM, **cm_kwargs)
simuResult = simulator.run(ifVerbose=False, pickN=None)
endTime = time.time()
print('  -Simulate T='+str(T)+' cost', round((endTime-startTime)/60, 3))

# currents -------------------------------------------
epsp_array = get_keyValues_2_2DArray(post_Obj,list(range(1)), T, dt, Key='EPSP') 
EPSP = pd.DataFrame(epsp_array).T; EPSP.columns = ['EPSP']

ipsp_array = get_keyValues_2_2DArray(post_Obj,list(range(1)), T, dt, Key='IPSP') 
IPSP = pd.DataFrame(ipsp_array).T; IPSP.columns = ['IPSP']

gE_array = get_keyValues_2_2DArray(post_Obj,list(range(1)), T, dt, Key='gE') 
gE = pd.DataFrame(gE_array).T; gE.columns = ['gE']

gI_array = get_keyValues_2_2DArray(post_Obj,list(range(1)), T, dt, Key='gI') 
gI = pd.DataFrame(gI_array).T; gI.columns = ['gI']

memP_array = get_keyValues_2_2DArray(post_Obj,list(range(1)), T, dt, Key='memPotential') 
memPotential = pd.DataFrame(memP_array).T; memPotential.columns = ['memPotential']

currents = pd.concat([EPSP, IPSP, gE, gI, memPotential ], axis=1); currents.to_csv(os.path.join(segments_path, 'currents.csv'))

# current feature summary
summary = { "mean_epsp": np.mean(EPSP), "std_epsp": np.std(EPSP),
            "max_epsp": np.max(EPSP), "min_epsp": np.min(EPSP),
            "skew_epsp": skew(EPSP), "kurtosis_epsp": kurtosis(EPSP),
            "entropy_epsp": entropy(np.histogram(EPSP, bins=20, density=True)[0] + 1e-10),
               
            "mean_ipsp": np.mean(IPSP), "std_ipsp": np.std(IPSP),
            "max_ipsp": np.max(IPSP), "min_ipsp": np.min(IPSP),
            "skew_ipsp": skew(IPSP), "kurtosis_ipsp": kurtosis(IPSP),
            "entropy_ipsp": entropy(np.histogram(IPSP, bins=20, density=True)[0] + 1e-10),
            
            "total_epsp": np.sum(EPSP), "total_ipsp": np.sum(IPSP),
            "epsp_ipsp_ratio": np.sum(EPSP) / (np.sum(IPSP) + 1e-8),
            "percent_zero_epsp": np.mean(EPSP < 1e-3), "percent_zero_ipsp": np.mean(IPSP < 1e-3) }
currentSummary = pd.DataFrame([summary]); currentSummary.to_csv(os.path.join(segments_path, 'currentSummary.csv'))


# spkTimes -------------------------------------------
spkTimes_array = get_keyValues_2_2DArray(post_Obj, list(range(1)), T, dt, Key='spkTimes') 
spkTimesPost_df = pd.DataFrame(spkTimes_array).T
spkTimesPost_df.columns = neuronName_post
spkTimesPost_df = spkTimesPost_df.loc[~(spkTimesPost_df==0).all(axis=1)]
print('  -post fired', len(spkTimesPost_df), 'times')

spkTimes_df = pd.concat([spkTimesPre_df, spkTimesPost_df], axis=1)
spkTimes_df.to_csv(os.path.join(segments_path, 'spkTimes.csv'))
print('  -spkTimes_df.shape:', spkTimes_df.shape)


# Getting General Parameters----------------------------
generalPara = {'T': T, 'dt':dt, 'num_pre':num_pre, 
               'per_preE':per_preE, 'num_preE':num_preE, 'per_preI':per_preI, 'num_preI':num_preI,
               'preBP_lb':args.preBP_lb, 'preBP_ub':args.preBP_ub,  
               'g_Leak':post_Obj[0].get('g_Leak'), 'tau_m':post_Obj[0].get('tau_m'), 'tref':post_Obj[0].get('tref'),
               'gE2E_bar':gE2E_bar, 'gI2E_bar':gI2E_bar,'tau_noise':tau_noise, 'mu_noise':mu_noise, 'sigma_noise':sigma_noise, }
with open(os.path.join(segments_path, "generalPara.json"), "w") as f:
    json.dump(generalPara, f, indent=4)
print("  -Done Saving generalPara to generalPara.json")


# =================================================================================================================
#                                              Feature Summary                                                    #
# =================================================================================================================
# neuron features---------------------------------------
cols=['BP', ]
paras_pre = extract_neuronParas_fromObjs(pre_Objs, keys=cols, neuronNames=neuronName_preE+neuronName_preI, ifSave=False)
paras_pre = paras_pre.apply(pd.to_numeric, errors="coerce"); paras_pre = paras_pre.reset_index()

ISI_list = []
for i in range(spkTimes_df.shape[1]):
    st = preprocess_spkTimes(spkTimes_df.iloc[:,i].values)
    _, _, burstingFraction, mean_ISI, _, _, cv_ISI, _= interSpikeIntervals_Stats(st,burstThreshold=burstThreshold, ifVerbose=False, ifPlotHist=False)
    real_fr = cal_firingRate(st, T, inDataType='spkTimes')
    feature_dic = {'real_FR':real_fr, 'firing_times': len(st),
                   'burstingFraction':burstingFraction,'cv_ISI': cv_ISI, 'mean_ISI': mean_ISI}
    ISI_list.append(pd.DataFrame.from_dict(feature_dic, orient='index'))
    
ISIs = pd.concat(ISI_list, axis=1).T.reset_index(drop=True)
ISIs['neuron_name'] = spkTimes_df.columns.to_list()

featureSummary = pd.merge(paras_pre, ISIs, on='neuron_name', how='outer')
featureSummary['weight'] = g_s
featureSummary.to_excel(os.path.join(segments_path, 'featureSummary.xlsx'))


# =================================================================================================================
#                                                   CCG                                                           #
# =================================================================================================================
postName = neuronName_post[0]
post_spkTimes = spkTimes_df[postName].dropna()
saveBin=True; cch_dic={}; normccg_dic={}; zccg_dic={}; jccg_dic={}; jzccg_dic={}; logccg_dic={}; dcCCH_dic={}

# 40 pre - 1 post (all connected pairs)
for i in range(num_pre):
    preName=neuronName_pre[i]
    pre_spkTimes = spkTimes_df[preName].dropna()
    g_bar = g_s[i]
    BP, rFR, BF, CV = featureSummary[featureSummary['neuron_name']==preName][['BP','real_FR', 'burstingFraction', 'cv_ISI']].values[0]
    title = preName+':BP='+str(round(BP,3))+'  rFR='+str(round(rFR, 3))+"  bursting%="+str(round(BF, 3))+"  CV%="+str(round(CV, 3))
    times, idx, nspks1, nspks2 = merge_spkTimes(pre_spkTimes, post_spkTimes)

    # CCG & Plot------------------------
    ccg_result = cal_CCG(times, idx, duration=duration, bin_size=bin_size, ifPlot=False, ifSave=False)
    cch, cch_diffs, ach1, ach2, n_bins, half_bins, t = ccg_result
    cch_dic[preName+'_'+postName] = np.concatenate((cch, [g_bar]))
    plot_ccg(ccg=cch, timeBins=t, title=title, ifSave=True, barColor=ccgColor,
             savePath=os.path.join(ccgResult_path, 'plots'), filename=preName+'-'+postName)

    if args.generateOtherCCGs:
        # other kinds of CCGs --------------
        all_ccgs = allKinds_CCG(cch, nspks1, nspks2, T, dt, pre_spkTimes, post_spkTimes, duration=duration, 
                                bin_size=bin_size, epsilon=1.0, n_surrogates=50, jitter_window_ms=20)
        norm_ccg, z_ccg, jitterCorrected_ccg, jitterCorrectedZscored_ccg, log_ccg = all_ccgs
        # dcCCH------------------------
        dccch = cal_dccch(cch, ach1, ach2, nspks1, nspks2, n_bins, half_bins, t, preName, postName,
                            featureSummary, ifPlot=False, ifSave=False)
        # summary --------------------------
        normccg_dic[preName+'_'+postName] = np.concatenate((norm_ccg, [g_bar]))
        zccg_dic[preName+'_'+postName] = np.concatenate((z_ccg, [g_bar]))
        jccg_dic[preName+'_'+postName] = np.concatenate((jitterCorrected_ccg, [g_bar]))
        jzccg_dic[preName+'_'+postName] = np.concatenate((jitterCorrectedZscored_ccg, [g_bar]))
        logccg_dic[preName+'_'+postName] = np.concatenate((log_ccg, [g_bar]))     
        dcCCH_dic[preName+'_'+postName] = np.concatenate((dccch, [g_bar]))

    if saveBin:
        info = {'n_bins':n_bins, 'half_bins':half_bins, 't':t }
        info_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in info.items() ]))
        info_df.to_csv(os.path.join(ccgResult_path, 'binInfo.csv'))
        saveBin=False

if args.generateUnconnectedCCG:
    for i in range(num_pre):
        j = np.random.choice([m for m in range(40) if m != i])

        preName_i=neuronName_pre[i]
        preName_j=neuronName_pre[j]
        pre_spkTimes_i = spkTimes_df[preName_i].dropna()
        pre_spkTimes_j = spkTimes_df[preName_j].dropna()
        
        g_bar = 0
        times, idx, nspks1, nspks2 = merge_spkTimes(pre_spkTimes_i, pre_spkTimes_j)

        # CCG & Plot------------------------
        ccg_result = cal_CCG(times, idx, duration=duration, bin_size=bin_size, ifPlot=False, ifSave=False)
        cch, cch_diffs, ach1, ach2, n_bins, half_bins, t = ccg_result
        cch_dic[preName_i+'_'+preName_j] = np.concatenate((cch, [g_bar]))

        if args.generateOtherCCGs:
            # other kinds of CCGs --------------
            all_ccgs = allKinds_CCG(cch, nspks1, nspks2, T, dt, pre_spkTimes_i, pre_spkTimes_j, duration=duration, 
                                    bin_size=bin_size, epsilon=1.0, n_surrogates=50, jitter_window_ms=20)
            norm_ccg, z_ccg, jitterCorrected_ccg, jitterCorrectedZscored_ccg, log_ccg = all_ccgs
            # dcCCH ----------------------------
            dccch = cal_dccch(cch, ach1, ach2, nspks1, nspks2, n_bins, half_bins, t, preName_i, preName_j,
                                featureSummary, ifPlot=False, ifSave=False)
            # summary --------------------------
            normccg_dic[preName_i+'_'+preName_j] = np.concatenate((norm_ccg, [g_bar]))
            zccg_dic[preName_i+'_'+preName_j] = np.concatenate((z_ccg, [g_bar]))
            jccg_dic[preName_i+'_'+preName_j] = np.concatenate((jitterCorrected_ccg, [g_bar]))
            jzccg_dic[preName_i+'_'+preName_j] = np.concatenate((jitterCorrectedZscored_ccg, [g_bar]))
            logccg_dic[preName_i+'_'+preName_j] = np.concatenate((log_ccg, [g_bar]))     
            dcCCH_dic[preName_i+'_'+preName_j] = np.concatenate((dccch, [g_bar]))


CCH_df = pd.DataFrame.from_dict(cch_dic, orient='index').T; CCH_df.to_csv(os.path.join(ccgResult_path, 'CCH.csv'))
if args.generateOtherCCGs:
    normCCH_df = pd.DataFrame.from_dict(normccg_dic, orient='index').T; normCCH_df.to_csv(os.path.join(ccgResult_path, 'normCCH.csv'))
    zCCH_df = pd.DataFrame.from_dict(zccg_dic, orient='index').T; zCCH_df.to_csv(os.path.join(ccgResult_path, 'zCCH.csv'))
    jCCH_df = pd.DataFrame.from_dict(jccg_dic, orient='index').T; jCCH_df.to_csv(os.path.join(ccgResult_path, 'jCCH.csv'))
    jzCCH_df = pd.DataFrame.from_dict(jzccg_dic, orient='index').T; jzCCH_df.to_csv(os.path.join(ccgResult_path, 'jzCCH.csv'))
    logCCH_df = pd.DataFrame.from_dict(logccg_dic, orient='index').T; logCCH_df.to_csv(os.path.join(ccgResult_path, 'logCCH.csv'))
    dcCCH_df = pd.DataFrame.from_dict(dcCCH_dic, orient='index').T; dcCCH_df.to_csv(os.path.join(ccgResult_path, 'dcCCH.csv'))

print('  -Done Saving CCG Realted.')



# =================================================================================================================
#                                             Examing Plots                                                       #
# =================================================================================================================
# E/I Current-------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(20, 5*2))
ax[0].plot(EPSP, c=mC, label='Ecurr')
ax[0].plot(IPSP, c=bC, label='Icurr')
ax[0].legend();ax[0].set_title("Ecurr/ICurr Over Time", fontweight='bold')
ax[1].plot(EPSP.loc[3000:3500], c=mC); ax[1].plot(IPSP.loc[3000:3500], c=bC); ax[1].set_title('E/Icurr it3000-3500')
ax[2].plot(gE, c='orange', label='gE')
ax[2].plot(-1*gI, c='k', label='gI')
ax[2].legend(); ax[2].set_title("gE/gI Over Time", fontweight='bold')
ax[3].plot(gE.loc[3000:3500], c='orange'); ax[3].plot(-1*gI.loc[3000:3500], c='k'); ax[3].set_title('gE/gI it3000-3500')
plt.tight_layout(); plt.savefig(os.path.join(segments_path, 'EICurrent'), bbox_inches="tight"); plt.close()


# memPotential------------------------------------------
spkTrin_Exc = spkTrainPre_df.filter(like='Pre_E', axis=1).sum(axis=1).values
spkTrin_Inh = spkTrainPre_df.filter(like='Pre_I', axis=1).sum(axis=1).values

sIdx=max(0, int(spkTimesPost_df.values[1]-100)); eIdx=int(spkTimesPost_df.values[0]+300)

plt.figure(figsize=(20, 3))
t_sp_exc = range_t[sIdx:eIdx][spkTrin_Exc[sIdx:eIdx] > 0.5]
t_sp_inh = range_t[sIdx:eIdx][spkTrin_Inh[sIdx:eIdx] > 0.5]
plt.plot(t_sp_exc, np.ones(len(t_sp_exc))*(-30), '|', c=mC, ms=20, markeredgewidth=3)
plt.plot(t_sp_inh, np.ones(len(t_sp_inh))*(-40), '|', c=bC, ms=20, markeredgewidth=3)
plt.plot(range_t[sIdx:eIdx], memPotential.values[sIdx:eIdx], c=ccgColor, lw=2)
plt.tight_layout(); plt.savefig(os.path.join(segments_path, 'memPotential'), bbox_inches="tight"); plt.close()


print('  -Done.')


