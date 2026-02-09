'''
# validation.py
# Author: Xiaoqian Sun, 06/12/2024
# 
'''


# Import Packages
#========================================================================================
import os 
import math
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')






# functions
#========================================================================================
def ckech_same_NeuronDynamics(neuronObjs, key, N):
    '''
    This is try to find if any neuronObjs showed same dynamics with any other neuron
    it's highly impossible, so, need to check
    E.g., result = sameNeuronDynamics(simulator.neurons, 'spkTimes', Ne, Ni)
          print(result)
    '''
    
    summary={}
    for i in range(N):
        key_i = np.array(neuronObjs[i].get(key))

        sameN = []
        for j in range(N):
            key_j = np.array(neuronObjs[j].get(key))

            if len(key_i)>0 and len(key_j)>0:
                if i != j and (key_i == key_j).all() :
                    sameN.append(j)
        summary['neuron'+str(i)] = sameN
        
    
    if summary==None:
        return ('Safe')  # no neuorns have same dyanmics with each other
    else:
        return(summary) # some neurons need attention and return the list 


def check_memPotentialCalc_1N(neuronObj):
    
    # extract values
    Nt = neuronObj.get('Nt')
    dt = neuronObj.get('dt')
    tref = neuronObj.get('tref')
    tau_m = neuronObj.get('tau_m')
    spkTimes = neuronObj.get('spkTimes') 
    EPSP = neuronObj.get('EPSP')
    IPSP = neuronObj.get('IPSP')
    LeakC = neuronObj.get('LeakC')
    Xcurr = neuronObj.get('Xcurr')
    memPotential = neuronObj.get('memPotential')
    # calculate dvs
    dvs = [0]+[memPotential[i]-memPotential[i-1] for i in range(1, Nt)]

    # form df
    neuronObj_summary = pd.DataFrame(np.stack((memPotential, np.array(dvs),EPSP, IPSP, Xcurr, LeakC), axis=1),
                                     columns=['memPotential', 'dv', 'EPSP', 'IPSP', 'Xcurr', 'LeakC'])
    
    # calculate dv_post
    neuronObj_summary['dv_post'] =  neuronObj_summary.iloc[:, 2:].sum(axis=1) * (dt/tau_m)
    
    
    # compare
    math_list = neuronObj_summary['dv_post'].apply(lambda x: round(x, 10)) == neuronObj_summary['dv'] .apply(lambda x: round(x, 10))
    unmatch_idx = np.where(math_list==False)[0]*dt

    # expend spkTimes:
    exp_spkTimes = []
    for spktime in spkTimes:
        exp_spkTimes.append(spktime)
        exp_spkTimes.append(spktime+tref)


    return(unmatch_idx == exp_spkTimes)

def check_memPotentialCalc(neuronObjs, N):
    
    '''
    check if the calculation of membrane potential was correct.
    
    Membrane potential was calculated in this way:
        LeakC = V-V_LeakReversal  /g_leak 
        Ecurr = gE * (V-V_excReversal) / g_Leak  
        Icurr = gE * (V-V_excReversal) / g_Leak 
        Xcurr = externalInput / g_Leak  
        dv = (LeakC + Ecurr + Icurr + Xcurr) * (dt / tau_m)
        v = v + dv
        
        so we can get dv by v[i+1] - v[i]
    
    Here, we use recorded LeakC/Ecurr/Icurr/Xcurr to calculate dv again (dv_post) and try to see if dv_post==dv
    Note that the calculation was a little bit different when a spike happened, so the next step is to see
    if unmatched idx are where a spike happened and ended.
    If so, we're safe
    If not, we need to zoom in
    '''
    
    flags = []
    for i in range(N):
        neuronObj = neuronObjs[i]
        flags.append(check_memPotentialCalc_1N(neuronObj))
        
    unmatch_neurons = np.where(np.array(flags)==False)[0]
    if len(unmatch_neurons) == 0:
        return('Safe')
    else:
        return(unmatch_neurons)


def check_valueRange(neuronObj, key, maxV=None, minV=None, initV=None):
    '''
    try to see if a given dynamics is in certain range
    e.g., if:
        - starts from initV
        - max <= maxV
        - min >= minV
    
    '''
    neuronDynamics = neuronObj.get(key)
    
    flags = []
    if initV:
        flags.append(neuronDynamics[0] == initV)

    if maxV: 
        flags.append(neuronDynamics.max() <= maxV)

    if minV: 
        flags.append(neuronDynamics.min() >= minV)
        
    
    if (np.array(flags)==True).all():
        return('Safe')
    else:
        return flags















