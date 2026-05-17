'''
# utils.py
# Author: Xiaoqian Sun, 07/03/2024
# utilities function
'''


# Import Packages
#========================================================================================
import os 
import re
import math
import pickle
import scipy as sp
import numpy as np
import pandas as pd
from tslearn import metrics
from scipy.stats import kstest
from scipy.signal import correlate
from scipy.stats import expon, entropy
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# from spycon.coninf import GLMCC, GLMPP
import matplotlib.pyplot as plt

import neuron, simulation
from visualization import *; from utils import *

import warnings
warnings.filterwarnings('ignore')



# functions
#========================================================================================
#-------------------------------------- General --------------------------------------#
def orderByNumber(neuron):
    '''
    used in 1_replicatesGeneration.py or oder column by number, not lexicographically
    i.e., "post1", "post10", "post2" instead of "post1", "post2", "post10"
    '''
    return int(re.search(r'\d+', neuron).group())

def calculate_g_bar(num_exc, num_inh, p_e2e=0.1, p_e2i=0.2, p_i2e=0.3, p_i2i=0.1, gBar_ratio=None, weight_ratio=None):
    """
    Calculate g_bar given total conductance constraints for excitatory and inhibitory neurons.
    
    Arguments:
        - num_exc: Number of excitatory neurons
        - num_inh: Number of inhibitory neurons
        - connect_probability: Probability of synaptic connections
    
    Returns: g_bar: Baseline synaptic conductance (nS)
    """
    
    # Fixed total conductance levels for a single neuron (nS)
    G_E2E_total, G_I2E_total = 20, 100  # exc neuron: total E→E and I→E conductance
    G_E2I_total, G_I2I_total = 40, 20   # inh neuron: total E→I and I→I conductance
    
    # Fixed g_bar ratio
    if type(gBar_ratio)==type(None):
        gBar_ratio = {'gE2E_bar':1., 'gE2I_bar':2., 'gI2E_bar':3., 'gI2I_bar':2.,}
    
    
    # Fixed weight ratio
    if type(weight_ratio)==type(None):
        weight_ratio = {'w_exc2exc':3., 'w_exc2inh':2., 'w_inh2exc':5., 'w_inh2inh':1., }
    
    # num of incoming synapses exc/inh
    e2e_synapses = int(p_e2e * num_exc); i2e_synapses = int(p_i2e * num_inh) 
    e2i_synapses = int(p_e2i * num_exc); i2e_synapses = int(p_i2i * num_inh)

    # g_bar baseline
    g_bar_e2e = G_E2E_total / (e2e_synapses * weight_ratio['w_exc2exc'] * gBar_ratio['gE2E_bar'])  # from E→E
    g_bar_i2e = G_I2E_total / (i2e_synapses * weight_ratio['w_inh2exc'] * gBar_ratio['gI2E_bar'])  # from I→E
    g_bar_e2i = G_E2I_total / (e2i_synapses * weight_ratio['w_exc2inh'] * gBar_ratio['gE2I_bar'])  # from E→I
    g_bar_i2i = G_I2I_total / (i2e_synapses * weight_ratio['w_inh2inh'] * gBar_ratio['gI2I_bar'])  # from I→I

    # take the average
    g_bar = np.mean([g_bar_e2e, g_bar_i2e, g_bar_e2i, g_bar_i2i])

    return g_bar



#---------------------------- histogram, exp distribution ----------------------------#
def histo_info(datapoints):
    
    '''get distribution info from a list of datapoints'''
    
    hist, bin_edges = np.histogram(datapoints, density=True)
    x_hist_middle = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(hist))]

    data_mean = np.mean(datapoints)
    

    return(hist, x_hist_middle, data_mean)

def calculate_exp(meanValue, x):
    
    '''given mean and x, generate exponential y values'''
    
    rate = 1/meanValue
    y_exp = rate * np.exp(-rate*np.array(x))

    return(y_exp)

def drawFrom_exp(meanValue, numPoints):
    '''randomly draw points from exponential distribution given mean and number of points'''
    
    datapoints = np.random.exponential(scale=meanValue, size=numPoints) 
    
    # gather histogram info of this array
    hist, x_middle, dataMean = histo_info(datapoints)
    
    return(datapoints, hist, x_middle, dataMean)
        
def compare_rmse(y_actual, y_fit):
    '''calculate the Root_Mean_Squared_Error between two arries'''
    
    rmse = np.sqrt(mean_squared_error(y_actual, y_fit))
    
    return(rmse)

def ks_test(data, ifVerbose=False):
    
    '''
    compare data with an exponential distribution
        - ks_statistic ranges [0,1]
            0 - perfect match
            Higher values indicate a greater difference 

        - p_value < 0.05, accept
    '''
    
    ks_statistic, p_value = kstest(data, 'expon', args=(0, np.mean(data)))
    if ifVerbose:
        print("KS Score:", round(ks_statistic,4), "| P-value:", round(p_value,4))

    
    return(ks_statistic, p_value)

def evaluate_FR_exp(firing_rates, histTitle=None, thresholdPer=0.2, ifPlot=True, ifVerbose=False):
    
    '''
    given firing rate, exam:
        - if the distribution of firing rates fits an exp
        - if histo bars is an exp distribution'''
    
    
    hist, x_m, mean = histo_info(firing_rates)
    
    # 1st if
    hist_exp = calculate_exp(mean, x_m)
    rmse = compare_rmse(hist, hist_exp)
    rmse_threshold = np.sum(hist*thresholdPer)
    if rmse < rmse_threshold and ifVerbose:
        print('rmse within', str(thresholdPer*100)+'% fluctuation. RMSE =', rmse)
        
    # 2nd if
    ks_statistic, p_value = ks_test(firing_rates, ifVerbose=ifVerbose)
    

    if ifPlot:
        # Plot the histogram and the fitted curve
        plt.figure(figsize=(4, 2))
        
        plt.hist(firing_rates, density=True, alpha=0.6, color='g')

        plt.plot(x_m, hist, c='k', label='Mid Bin Connect')
        plt.plot(x_m, hist_exp, 'r-', lw=2, alpha=0.6, label='Fitted Exponential Curve')
        
        plt.title(histTitle+' RMSE='+str(round(rmse, 6))); plt.ylabel('Density'); plt.legend(loc=0); plt.show()
            
    
    return(rmse, ks_statistic, p_value)


#------------------------------------- sampling -------------------------------------#
def sample_gaussian(sampleSize, mu, sigma, lower_bound, upper_bound):

    '''
    draw randomly from gaussian distribution, if range truncated to [LB, UB]
    '''
    from scipy.stats import truncnorm
    
    # standar normal bounds
    sLB, sUB = (lower_bound-mu)/sigma, (upper_bound-mu)/sigma, 

    # draw samples
    samples = truncnorm.rvs(sLB, sUB, loc=mu, scale=sigma, size=sampleSize)

    return samples






#-------------------------- firing rate, F-I curve, Rheobase --------------------------#
def cal_firingRate(inputData, T, inDataType='spkTrain'):
    
    '''
    calculate firing rate (HZ), spikes/second
    
    Argument:
        - spkTrain: 1D/2D array, if 2D array, the input spkTrain should be in shape (num_neuron, num_timesteps)
        - duration: in timeUnit (ms/s)
    '''

    
    if inDataType == 'spkTrain':
    
        inputData = np.asarray(inputData)
        if len(inputData.shape)==1:
            numSpikes = inputData.sum()
            numSpk_array = np.asarray([numSpikes])
        else:        
            numSpk_array = np.asarray([inputData[i].sum() for i in range(inputData.shape[0])])
            
        return(numSpk_array/(T/1000))
    
    elif inDataType == 'spkTimes':
        inputData = preprocess_spkTimes(inputData)
        return (len(inputData)/(T/1000))
    else:
        raise ValueError('Input must be either spkTrain or spkTimes.')

def get_firingRates(neuronObj_list, neuron_idx, T, dt, spkKey='spkTrain'):
    '''
    given neuron idx, extract spk train of that neuron and calcualte firing rate
    2 options for spkKey: 'spkTrain', 'assignSpkTrain'
    '''
    
    spkTrains = get_keyValues_2_2DArray(neuronObj_list, neuron_idx, T, dt, Key=spkKey)
    FR_array = cal_firingRate(spkTrains, T, inDataType='spkTrain')
        
    return(FR_array)

def get_keyValues_2_2DArray(neuronObj_list, neuron_idx, T, dt, Key='spkTrain'):
    '''
    given neuron idx, extra data to 2D array for further analysis
    note that the output shape will be (num_neurons, num_datapoints)
    data can be formed into 2D array including: 
        'assignSpkTrain', 'spkTrains', 'memPotential', 'synTrace'
        'gE', 'gI', 'EPSP', 'IPSP', 'Xcurr', 'LeakC', 'spkTimes'
    '''

    N = len(neuron_idx)
    Lt = int(T/dt)
    outputArray = np.zeros((N, Lt))

    neuronIdx = 0
    for i in neuron_idx:
        neuronObj = neuronObj_list[i]
        data = neuronObj.get(Key)
        outputArray[neuronIdx][0:len(data)] = data
        neuronIdx+=1

    return(outputArray)

def loop_xInput_simulation(N_kwargs, T, dt, gE_bar, gI_bar, neuronType, N, Ne, Ni, maxns, connectivityM, startC, endC, numC, step=None):
    '''
    convinently adjust startC and endC and get reasonable F-I-curve
    '''
    
    if type(numC) != type(None):
        currents = np.linspace(startC, endC, numC).astype('int')
    elif type(step) != type(None):
        currents = list(range(startC, endC, step))
    else:
        raise ValueError('numC and step can not both be None to generate current list')
    
    FRs = []; numSpikes = []
    for I_b in currents:

        N_kwargs['externalInput'] = I_b
        N_obj = neuron.Neuron(T, dt, gE_bar, gI_bar, neuronType, **N_kwargs)

        simulator = simulation.Simulator(T, dt, N, Ne, Ni, maxns, [N_obj], CM=connectivityM)
        simulator.run(ifVerbose=False, pickN=None)    

        FRs.append(cal_firingRate(N_obj.get('spkTrain'), T, inDataType='spkTrain')[0])
        numSpikes.append(simulator.get('ns'))
        
    return(currents, FRs)

def check_spkOccurPosition(currents, FRs, startC, endC):
    
    '''
    check spike position and adjust current range accordingly
    '''
    
    ifChange = False
    
    spikePosition = np.where(np.array(FRs) != 0)[0]

    if len(spikePosition) == 0:
        # no spikes for all current
        startC = endC
        endC = endC + 200
        ifChange = True
    elif len(spikePosition) == len(FRs):
        startC = max(startC-200, 0)
        endC = startC+200
        ifChange = True
    else:
        spikeStartPosition = spikePosition[0] 

        if spikeStartPosition/len(FRs) > 0.75:
            startC = int(np.percentile(currents, 50))
            endC = startC  + 200
            ifChange = True
        elif spikeStartPosition/len(FRs) < 0.25:
            startC = int(startC * 0.75)
            endC = startC  + 200
            ifChange = True
            
    return (ifChange, startC, endC)

def linear_regression(x, y):
    
    slope, intercept, r, p, std_err = sp.stats.linregress(x, y)
    fit_y = np.array(x) * slope + intercept 
    
    return (slope, intercept, fit_y)

def F_I_curve(N_kwargs, T, dt, N, Ne, Ni, maxns, neuronType=0, startC=100, endC=300, numC=20, step=None, ifPlot=False, ifSave=False, savePath=None, filename=None):
    
    '''
    input various input current and get the firing rate – VS – input current curve
    which helps to get an idea of the neuron sensitivity to exc input
    also help to find Rheobase value of a neuron
    '''

    gE_bar, gI_bar = 0, 0
    connectivityM = np.zeros((N, N))
    
    
    # using while loop to find desired current range to generate firing rates with some 0s and some FRs
    ifChange = True
    while ifChange:
        # loop and get FRs
        currents, FRs = loop_xInput_simulation(N_kwargs, T, dt, gE_bar, gI_bar, neuronType, N, Ne, Ni, maxns, 
                                               connectivityM, startC=startC, endC=endC, numC=numC, step=None)
        # check
        ifChange, startC, endC = check_spkOccurPosition(currents, FRs, startC, endC)
        print('ifChange =', ifChange, 'current in range ['+str(startC)+', '+str(endC)+']')
        
    
    # calculate slope
    spikeStartPosition = np.where(np.array(FRs) != 0)[0][0]
    LR_startPosition = spikeStartPosition - 1

    nonZero_FRs = FRs[LR_startPosition:]
    nonZero_currents = currents[LR_startPosition:]
    
    slope, intercept, fit_FRs = linear_regression(nonZero_currents, nonZero_FRs)
    
        
    if ifPlot:
        fig, ax = plt.subplots(1,1, figsize=(5, 3))
        ax.plot(currents, FRs, '--o', color='b')
        ax.plot(nonZero_currents, fit_FRs, color='orange')
        ax.set_xlabel('Current (pA)');ax.set_ylabel('Spikes/sec')
        ax.spines[['right', 'top']].set_visible(False)

        
    # save/show
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()
        
        
    return(currents, FRs, slope, spikeStartPosition)

def find_Rheobase(N_kwargs, T, dt, N, Ne, Ni, maxns, searchStart, searchEnd, neuronType=0):
    '''
    by using F-I curve, we try to fine search Rheobase
    the minimum constant current amplitude required to depolarize a neuron to the threshold for generating an action potential. 
    '''
    
    gE_bar, gI_bar = 0, 0
    connectivityM = np.zeros((N, N))
    
    current, FRs= loop_xInput_simulation(N_kwargs, T, dt, gE_bar, gI_bar, neuronType, N, Ne, Ni, maxns,connectivityM, 
                                         startC=searchStart, endC=searchEnd, numC=None, step=1)
    Rheobase = current[np.where(np.array(FRs) !=0 )[0][0]]
    
    return(Rheobase)





#------------------------------------ ISIs related ------------------------------------#
def preprocess_spkTimes(spkTimes):
    '''
    remove any extra 0s/nans from spkTimes array (besides first 0)
    '''
    if len(spkTimes) < 2:
        raise ValueError ('Input spkTimes must contain a least 2 spikes.') 
    

    if spkTimes[0] == 0:
        spkTimes = [spkTimes[0]] + [x for x in spkTimes[1:] if x != 0]
    elif math.isnan(spkTimes[0]):
        raise ValueError ('Input spkTimes does not have any valid timepoints.') 
    else:
        spkTimes = [x for x in spkTimes if x != 0]
        spkTimes = [x for x in spkTimes if not math.isnan(x)]


    return list(spkTimes)

def spkTrain_2_spkTimes(spkTrain, dt):
    '''
    convert spkTrain into spkTimes in ms
    '''
    
    spike_indices = np.where(spkTrain == 1)[0]
    spkTimes = spike_indices * dt

    return spkTimes

def cal_interSpikeIntervals(spkTimes):
    '''calculate the time intervals between consecutive spikes'''
    

    spkTimes = preprocess_spkTimes(spkTimes)
        
    InterSpikeIntervals = []
    for i in range(len(spkTimes)-1):
        InterSpikeIntervals.append(round(spkTimes[i+1]-spkTimes[i], 4))
        
    return(InterSpikeIntervals)

def detect_spkTrain_burst(ISIs, spkTimes, burstThreshold=5, ifVerbose=True):
    '''
    If there is a spikes in a row and the interval between each spike is smaller than burstThreshold, then we call this a burst.
    E.g., ISIs = [9.9, 1.73, 2.06, 1.47, 1.53, 1.72, 2.04, 1.65...]
    
    The return is a list of sublists, sublist contains the start and end index of ISIs and spkTimes
    E.g., 
        - burstPeriod: [[3, 13], [17, 23]]
        - burst in spkTrain: [[3, 14], [17, 24]]: [3, 14] means from the 3th spike to 14th spike, including 14th, these are in burst period
    '''
    if ifVerbose:
        print('ISIs:', ISIs)
    burst_starts=[]; burst_ends=[]; inside_burst = False
    for i in range(len(ISIs)):
        if ISIs[i] < burstThreshold:
            if not inside_burst:
                burst_starts.append(i)
                inside_burst = True
        else:
            if inside_burst:
                burst_ends.append(i - 1)
                inside_burst = False
    
    # if burst reaches the end of the spkTrain
    if inside_burst:
        burst_ends.append(len(ISIs) - 1)
    
    # burst index in ISI or spkTrain
    burstPeriod = []; spkPerids = []
    for start, end in zip(burst_starts, burst_ends):
        if end-start > 2: # more than 3 spks in a row
            burstPeriod.append([start, end])
            spkPerids.append([start, end+1])
    
            if ifVerbose:
                print(f"  -Burst from ISI {start} to {end}: {ISIs[start:end+1]}")
                print(f"  -Burst from spkTimes {start} to {end+1}: {spkTimes[start:end+2]}")
                print('  ---------------------------------------------------------------------')
            
    return(burstPeriod, spkPerids)

def cal_nonBursty_firingRate(spkTimes, T, spkBurstPerids):

    '''
    only calculate firing rate in non-bursty period
    that is exclude bursty period from spkTrain and then use remaining T and spkTrain to calculate firing rate
    this function only process 1 spkTimes array (associated with the spkBurstPeriods for this spkTimes array)
    '''
    remainingT = T
    non_bursty_spkTimes = spkTimes.copy()
    
    for burstSpk in spkBurstPerids:
    
        periodTime = spkTimes[burstSpk[1]] - spkTimes[burstSpk[0]]
        remainingT -= periodTime
        
        period_spkTimes = spkTimes[burstSpk[0]:burstSpk[1]+1]
        indices = np.where(np.isin(non_bursty_spkTimes, period_spkTimes))
        non_bursty_spkTimes = np.delete(non_bursty_spkTimes, indices)
    
    non_bursty_fr = len(non_bursty_spkTimes)/(T/1000)
    return(non_bursty_fr)
        
def calculate_burstFraction(spkTimes, burstTresh=7):
    '''
    level of burstness of the spkTrain, calculated as BF = num_burst_spike / total_num_spikes
    '''

    ISIs = cal_interSpikeIntervals(spkTimes)

    burst_ISIs = sum(isi < burstTresh for isi in ISIs)
    BF = burst_ISIs / (len(ISIs)+1) if len(ISIs) > 0 else 0.0

    return round(BF, 3)

def interSpikeIntervals_Stats(spkTimes, burstThreshold=7, ifVerbose=True, ifPlotHist=True):
    
    '''
    std: indicates the variability in Inter-Spike-Intervals
    cv: coefficient of variation (CV = Std / Mean) quantifies the relative variability of ISIs
        - higher CV indicates more irregular firing
        - cv = 0: no variability
        - cv < 1: low, regular firing, second-order gamma distribution (k=1 with cs~0.71)
        - cv ≈ 1: moderate, Poisson-like
        - cv > 1: high, irregular
        
    serial correlation :
        - positive: long ISIs are likely to follow long ISIs, short ISIs follow short ISIs
        - negative: long ISIs are followed by short ISIs and vice versa
        - 0: no predictable pattern between consecutive ISIs (which is the case we want)
    '''

    # first, get ISIs
    #-------------------------------------------------------------
    InterSpikeIntervals = cal_interSpikeIntervals(spkTimes)

    # stats
    #-------------------------------------------------------------
    mean_ISI = np.mean(InterSpikeIntervals)
    median_ISI = np.median(InterSpikeIntervals)
    std_ISI = np.std(InterSpikeIntervals)
    cv_ISI = std_ISI / mean_ISI 
    serial_CC = np.corrcoef(InterSpikeIntervals[:-1], InterSpikeIntervals[1:])[0, 1]

    # burst
    #-------------------------------------------------------------
    burstPeriod, spkPerids = detect_spkTrain_burst(InterSpikeIntervals, spkTimes, burstThreshold=burstThreshold, ifVerbose=ifVerbose)
    burstingFraction = calculate_burstFraction(spkTimes, burstTresh=burstThreshold)

    # pirnt out
    #-------------------------------------------------------------
    if ifVerbose:
        print()
        print('Bursting fraction =', burstingFraction)
    
    
    if ifVerbose:
        print()
        if cv_ISI>=0 and cv_ISI<0.2:
            print('CV = '+str(round(cv_ISI, 4))+'. No variability in Inter-Spike-Intervals.')
        elif cv_ISI<0.8 and cv_ISI>=0.2: 
            print('CV = '+str(round(cv_ISI, 4))+'. Low variability in Inter-Spike-Intervals.')
        elif cv_ISI<1 and cv_ISI>=0.8:
            print('CV = '+str(round(cv_ISI, 4))+'. (Moderate)Poisson-like variability in Inter-Spike-Intervals.')
        else:
            print('CV = '+str(round(cv_ISI, 4))+'. (High)Poisson-like variability in Inter-Spike-Intervals.')
            
    if ifVerbose:
        print()
        if serial_CC == 0:
            print('serial correlation = 0. No Predicatble pattern between consecutive ISIs.')
        elif serial_CC > 0:
            print('serial correlation = '+str(round(serial_CC, 4))+'. Long Spks Follow Short Ones.')
        else:
            print('serial correlation = '+str(round(serial_CC, 4))+'. Long Spks Follow Long Ones.')

        
    if ifPlotHist:
        plt.figure(figsize=(4, 2))
        plt.hist(InterSpikeIntervals)
        plt.xlabel('Inter-Spike Interval (ms)'); plt.ylabel('Frequency'); plt.title('ISIs histogram'); plt.show()


    return (burstPeriod, spkPerids, burstingFraction, mean_ISI, median_ISI, std_ISI, cv_ISI, serial_CC)

def evenly_selectFromArray(input_array, numElement):
    '''evenly select from an array given number of elements to select from it'''

    step = len(input_array)//numElement
    select_array = input_array[::step][:numElement]

    return(select_array)

def downSample_burstSpk(spkTimes, spkTrain, T, range_t, fr_method='mean', selectMethod='random', burstThreshold=5, ifVerbose=False, ifPlot=False):

    '''
    The idea is that in burst period, randomly select # number of spikes, which is calculated based on the whole spkTrain's 
    firing rate, out of the total num spikes.    
    Steps:
        - calculate firing rate of this spkTrain
        - loop through each burst period
            - calculate how many spikes could happen given firing rate
            - randomly/evenly downsampled certain spikes and disgard the rest

    Note that 2 methods can be used for downsampling
        - random: np.random.choice
        - even: evenly_selectFromArray(input_array, numElement)
    '''
    InterSpikeIntervals = cal_interSpikeIntervals(spkTimes)
    burstPeriod, spkBurstPerids = detect_spkTrain_burst(InterSpikeIntervals, spkTimes, burstThreshold=burstThreshold, ifVerbose=ifVerbose)
    

    if fr_method == 'mean':
        fr = cal_firingRate(spkTrain, T, inDataType='spkTrain')[0]
    elif fr_method == 'non_bursty':
        fr = cal_nonBursty_firingRate(spkTimes, T, spkBurstPerids)
    else:
        raise ValueError("Choose firing rate calculation method from ['mean', 'non_bursty']")

    
    spkTimes_downSampled = spkTimes.copy()
    # loop through
    for burstSpk in spkBurstPerids:
        
        spkStartTime = spkTimes[burstSpk[0]]
        spkEndTime = spkTimes[burstSpk[1]]
        periodTime = spkEndTime - spkStartTime
    
        avgSpkNum = round(fr * periodTime /1000)
        oriSpkNum = len(spkTimes[burstSpk[0]:burstSpk[1]+1])
        if avgSpkNum < burstSpk[1]+1 - burstSpk[0]:

            if selectMethod == 'random':
                keepSpks = np.random.choice(spkTimes[burstSpk[0]:burstSpk[1]+1], avgSpkNum, replace=False)
                keepSpks.sort()
            elif selectMethod == 'even':
                keepSpks = evenly_selectFromArray(spkTimes[burstSpk[0]:burstSpk[1]+1], avgSpkNum)
            else:
                raise ValueError("Choose select method from ['random', 'even']")

            spkTimes_downSampled[burstSpk[0]:burstSpk[0]+avgSpkNum] = keepSpks
            spkTimes_downSampled[burstSpk[0]+avgSpkNum:burstSpk[1]+1] = 0
        
            if ifVerbose:
                print('Removing Spikes: ')
                print('  Based on '+fr_method+' firing rate, there should be', avgSpkNum, 'but now have', oriSpkNum, 'spikes')
                print('  Original spkTimes period:', spkTimes[burstSpk[0]:burstSpk[1]+1])
                print('  Downsamples spkTimes periodL:', keepSpks)
                print('  ---------------------------------------------------------------------')
        else:
            if ifVerbose:
                print("  Skip: The burst period contains fewer spikes than what would be expected based on avg firing rate.")
                print('  ---------------------------------------------------------------------')
        
    spkTimes_downSampled = spkTimes_downSampled[spkTimes_downSampled>0]


    # get spkTrain based on spkTimes
    spkTrain_downSampled = np.zeros_like(range_t)
    spike_indices = np.searchsorted(range_t, spkTimes_downSampled)
    spkTrain_downSampled[spike_indices] = 1
    
    
    # plot
    if ifPlot:
        plt.figure(figsize=(20, 1))
        plt.plot(spkTimes_downSampled, 1*np.ones(len(spkTimes_downSampled)), '|', color='orange', ms=20, markeredgewidth=2, label='downSampled')
        plt.plot(spkTimes, 2*np.ones(len(spkTimes)), '|', color='b', ms=20, markeredgewidth=2, label='original')
        plt.legend(loc=0); plt.show()

    return(spkTrain_downSampled, spkTimes_downSampled)

def spkTrain_df_burstProcess(spkTrain_df, T, range_t, cols=[], selectMethod='random', burstThreshold=5, ifVerbose=False, ifPlot=False, savePath=None, fileName=None):

    '''
    perform downSample_burstSpk() to each spkTrain in a dataframe, and save for later use
    '''


    
    if len(cols) == 0:
        target_cols = spkTrain_df.columns.tolist()
    else:
        target_cols = cols


    downSampled_spkTrain_df = spkTrain_df.copy()
    for target in target_cols:
        t_spkTrain = spkTrain_df[target].values
        t_spkTimes = range_t[t_spkTrain > 0.5]
    
        t_spkTrain_ds, t_spkTimes_ds = downSample_burstSpk(t_spkTimes, t_spkTrain, T, range_t, 
                                                           selectMethod=selectMethod, burstThreshold=burstThreshold, 
                                                           ifVerbose=ifVerbose, ifPlot=ifPlot)
        downSampled_spkTrain_df[target] = t_spkTrain_ds

    # save
    downSampled_spkTrain_df.to_csv(os.path.join(savePath, fileName))




#------------------------------------- CCG / dcCCH / GLMCC ------------------------------------#
def merge_spkTimes(pre_spkTimes, post_spkTimes):
    
    '''
    This function process pre/post spkTimes as input to CCG/GLMCC
        - merge pre/post spkTimes together
        - sort by time
        - sort neuron idx by time as well
    Return:
        - sorted times in ms
        - sorted idx
    
    '''
    
    # preprocess
    pre_spkT = preprocess_spkTimes(pre_spkTimes); pre_idx=[1]*len(pre_spkT)
    post_spkT = preprocess_spkTimes(post_spkTimes); post_idx=[2]*len(post_spkT)
    nspks1 = len(pre_spkT); nspks2 = len(post_spkT)  # num of spikes in pre/post-spkTrain

    # merge
    times = np.array(pre_spkT + post_spkT); idx = np.array(pre_idx + post_idx)
    
    if len(times) != len(idx):
        raise ValueError("The lens of 'spkTimes' and 'neuronIDs' must match")

    # sort, unique idx (1, 2)
    sort_idx = np.argsort(times); times = times[sort_idx]; idx = idx[sort_idx]


    return (times, idx, nspks1, nspks2)

def cal_CCG(times, idx, duration, bin_size, ifPlot=False, figsize=(6, 5), ifSave=False, savePath=None, filename=None):
    '''
    calculate CCG from spkTrain with refernece FMAToolbox/Analyses/CCG.m   
    this function applies to a pair of pre-post spkTimes, if apply to multiple pairs, call function in for loops 
    see tech details in `InferConnectivity/2-0-CCG.ipynb`
    '''

    
    # WIN - used in GLMCC CCG - half duration 
    WIN = int(duration/2)
    unique_idx = np.unique(idx); n_idx = len(unique_idx)

    # get num of bins
    half_bins = int(np.round(duration / (2 * bin_size))); n_bins = 2 * half_bins + 1
    t = np.arange(-half_bins, half_bins + 1) * bin_size
    
    # compute
    ccg = np.zeros((n_bins, n_idx, n_idx))
    for i, id1 in enumerate(unique_idx):
        for j, id2 in enumerate(unique_idx):
            
            '''
            Note:
                if i==j, it's calculating auto-correlogram
                case i=1, j=2 is the same as i=2, j=1
            '''
            if i > j:  # symmetric
                ccg[:, i, j] = ccg[:, j, i] 
                continue

            # time difference
            times1 = times[idx == id1]; times1 = times1.astype(np.float32)
            times2 = times[idx == id2]; times2 = times2.astype(np.float32)
            
            diffs = times2[:, None] - times1[None, :]; diffs = diffs.flatten()
            if i==0 and j==1:
                diffs_win = diffs[np.abs(diffs) <= WIN] # for GLMCC

            # binning
            bin_edges = np.arange(-half_bins - 0.5, half_bins + 1.5) * bin_size
            hist, _ = np.histogram(diffs, bins=bin_edges)

            # store
            ccg[:, i, j] = hist
            
    
    if ifPlot:
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        ax[0].plot(t, ccg[:, 0, 0], label='Pre', lw=2)
        ax[0].plot(t, ccg[:, 1, 1], label='Post', lw=2); ax[0].set_title('ACH'); ax[0].legend()
        ax[1].bar(t, ccg[:, 0, 1], width=1, color='peru', alpha=0.8, edgecolor='none'); ax[1].set_title('CCH-pre-post') 
        plt.tight_layout()
        
        
        if ifSave:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(os.path.join(savePath, filename))
            plt.close()
        else:
            plt.show()


    cch = ccg[:, 0, 1]       # cross-correlation histogram
    cch_diffs = diffs_win
    ach1 = ccg[:, 0, 0]      # auto-correlation for trigger train (pre-spkTrain)
    ach2 = ccg[:, 1, 1]      # auto-correlation for referred train (post-spkTrain)
    
    return (cch, cch_diffs, ach1, ach2, n_bins, half_bins, t)

def allKinds_CCG(raw_ccg, nspks1, nspks2, T, dt, spkTimes_i, spkTimes_j, duration, bin_size, 
                 epsilon=1.0, n_surrogates=200, jitter_window_ms=20):
    '''
    this code work on raw ccg and get:
        - firing rates normalized ccg (raw_ccg / fr1*fr2)
        - z-scored ccg
        - jitter-corrected ccg (devided by std or not)
        - log_norm ccg
    
    '''
    
    # normalize using FRs and duration --------------------
    fr1, fr2 = nspks1/(T/1000), nspks2/(T/1000)
    norm_ccg = raw_ccg.copy() / (fr1*fr2)
    
    # z-scored --------------------------------------------
    spkTimes_i = preprocess_spkTimes(spkTimes_i)
    spkTimes_j = preprocess_spkTimes(spkTimes_j)
    bins = np.arange(0, T+dt, dt)
    spkTrain_1, _ = np.histogram(spkTimes_i, bins=bins)
    spkTrain_2, _ = np.histogram(spkTimes_j, bins=bins)

    mu = (nspks1 * nspks2) / (T/dt)
    std = max((np.dot(spkTrain_1, spkTrain_1) - mu**2) * (np.dot(spkTrain_2, spkTrain_2) - mu**2), 0)
    sigma = np.sqrt(std)
    z_ccg = (raw_ccg.copy() - mu) / (sigma + 1e-9)

    
    # jitter-corrected -----------------------------------
    jitter_ccgs = []
    for i in range(n_surrogates):
        jittered_spkTimes_j = spkTimes_j + np.random.uniform(low = -jitter_window_ms, 
                                                             high = jitter_window_ms, size = len(spkTimes_j))

        # keep only jittered spikes inside recording time
        jittered_spkTimes_j = jittered_spkTimes_j[(jittered_spkTimes_j >= 0) & (jittered_spkTimes_j <= T)]

        # jittered CCG
        times, idx, nspks1, nspks2 = merge_spkTimes(spkTimes_i, jittered_spkTimes_j)
        jitter_ccg,_,_,_,_,_,_ = cal_CCG(times, idx, duration=duration, bin_size=bin_size, ifPlot=False, ifSave=False)
        jitter_ccgs.append(jitter_ccg)

    jitter_ccgs = np.array(jitter_ccgs)
    jitter_mean = np.mean(jitter_ccgs, axis=0)
    jitter_std = np.std(jitter_ccgs, axis=0)

    jitterCorrected_ccg = raw_ccg.copy() - jitter_mean
    jitterCorrectedZscored_ccg = jitterCorrected_ccg / (jitter_std+1e-9)


    # log -----------------------------------------------
    log_ccg = np.log1p(raw_ccg.copy() / epsilon)

    # return norm_ccg, z_ccg, log_ccg
    return norm_ccg, z_ccg, jitterCorrected_ccg, jitterCorrectedZscored_ccg, log_ccg

# updated 09/30/2025, not used yet
def allKinds_CCG_new(raw_ccg, nspks1, nspks2, T, dt, spkTimes_i, spkTimes_j, duration, bin_size, 
                 epsilon=1.0, n_surrogates=200, jitter_window_ms=20):
    '''
    Returns:
      norm_ccg  -> counts / expected_counts_under_independence  (unitless, invariant to T & bin)
      z_ccg     -> (counts - expected) / sqrt(expected)        (independence z-score per bin)
      jitterCorrected_ccg, jitterCorrectedZscored_ccg, log_ccg -> unchanged
    Notes:
      - T and bin_size are in **milliseconds**
      - nspks1 = # spikes in pre train, nspks2 = # spikes in post train
    '''
    import numpy as np

    # ---------- expected counts under independence (per lag bin) ----------
    # rates (Hz): r = N / T_sec
    T_sec = T / 1000.0
    Δ_sec = bin_size / 1000.0
    r1 = nspks1 / T_sec
    r2 = nspks2 / T_sec
    expected_per_bin = r1 * r2 * T_sec * Δ_sec  # E[C_k] = r1 * r2 * T * Δ

    # ---------- normalized CCG (ratio to independence) ----------
    # unitless; =1 means at expectation; >1 excess coincidences; <1 deficit
    norm_ccg = raw_ccg.copy() / (expected_per_bin + 1e-12)

    # ---------- z-scored CCG (independence z) ----------
    # Per-bin z: (C - E) / sqrt(E); variance ≈ E for Poisson coincidences
    z_ccg = (raw_ccg.copy() - expected_per_bin) / np.sqrt(expected_per_bin + 1e-12)

    # ---------- jitter-corrected (unchanged) ----------
    spkTimes_i = preprocess_spkTimes(spkTimes_i)
    spkTimes_j = preprocess_spkTimes(spkTimes_j)

    jitter_ccgs = []
    for _ in range(n_surrogates):
        jittered_spkTimes_j = spkTimes_j + np.random.uniform(
            low=-jitter_window_ms, high=jitter_window_ms, size=len(spkTimes_j)
        )
        # keep only jittered spikes inside recording time
        jittered_spkTimes_j = jittered_spkTimes_j[(jittered_spkTimes_j >= 0) & (jittered_spkTimes_j <= T)]

        # jittered CCG
        times, idx, nspks1_j, nspks2_j = merge_spkTimes(spkTimes_i, jittered_spkTimes_j)
        jitter_ccg, _, _, _, _, _, _ = cal_CCG(times, idx, duration=duration, bin_size=bin_size,
                                               ifPlot=False, ifSave=False)
        jitter_ccgs.append(jitter_ccg)

    jitter_ccgs = np.array(jitter_ccgs)
    jitter_mean = np.mean(jitter_ccgs, axis=0)
    jitter_std  = np.std(jitter_ccgs, axis=0)

    jitterCorrected_ccg         = raw_ccg.copy() - jitter_mean
    jitterCorrectedZscored_ccg  = jitterCorrected_ccg / (jitter_std + 1e-9)

    # ---------- log (unchanged) ----------
    log_ccg = np.log1p(raw_ccg.copy() / epsilon)

    return norm_ccg, z_ccg, jitterCorrected_ccg, jitterCorrectedZscored_ccg, log_ccg


def ccg_moveUpScale(ccg):
    
    ''' in range 0-1 '''
    
    shifted = ccg - ccg.min()
    return shifted / (shifted.max() + 1e-9)
    
def cal_dccch(cch, ach1, ach2, nspks1, nspks2, n_bins, half_bins, t, preName, postName,
              featureSummary, ifPlot=False, ifSave=False, savePath=None, filename=None):
    
    '''
    calculate dc-CCG from ccg with refernece EranStarkLab/CCH-deconvolution/blob/main/cchdeconv.m  
    this function applies to a pair of pre-post spkTimes, if apply to multiple pairs, call function in for loops 
    see tech details in `InferConnectivity/2-1-dcCCH.ipynb`
    '''
    

    # make sure inputs are np.array
    cch, ach1, ach2 = map(np.asarray, (cch, ach1, ach2))
    nspks1, nspks2 = map(int, (nspks1, nspks2))
    
    # valide shape
    if cch.shape != ach1.shape or cch.shape != ach2.shape:
        raise ValueError("CCH, ACH1, and ACH2 must have the same shape.")
    if cch.shape[0] % 2 == 0:
        raise ValueError("CCH must have an odd number of bins.")

    
    # preprocess ach1
    ach1_normed = (ach1 - ach1.mean()) / nspks1 # extract mean + normalization
    ach1_normed[half_bins] = 1 - np.sum(ach1_normed[np.arange(n_bins) != half_bins])
    
    # preprocess ach2
    ach2_normed = (ach2 - ach2.mean()) / nspks2 # extract mean + normalization
    ach2_normed[half_bins] = 1 - np.sum(ach2_normed[np.arange(n_bins) != half_bins])
    
    # deconvolution
    # time domain to frequency domain 
    fft_ach1 = np.fft.fft(ach1_normed); freqs_ach1 = np.fft.fftfreq(len(ach1_normed))
    fft_ach2 = np.fft.fft(ach2_normed); freqs_ach2 = np.fft.fftfreq(len(ach2_normed))
    fft_cch = np.fft.fft(cch); freqs_cch = np.fft.fftfreq(len(cch))
    
    # remove achs effect & inverse
    den = fft_ach1 * fft_ach2 
    dccch = np.fft.ifft(fft_cch / den).real 
    dccch = np.roll(dccch, -1); dccch[dccch < 0] = 0

    if ifPlot:
        plot_dcCCH_process(preName, postName, ach1, ach1_normed, freqs_ach1, fft_ach1, 
                           ach2, ach2_normed, freqs_ach2, fft_ach2, 
                           cch, freqs_cch, fft_cch, dccch, t, featureSummary, ifSave=ifSave, savePath=savePath, filename=filename)


    return dccch    

def TP_FN_Accc(df, pred_colName, gt_colName):
    '''
    calculate TP/FP/TN/FN/precision/recall/f1_score
    the df should contain at lease 2 cols: prediction, groud truth label
    '''
    
    # Calculate TP, FP, TN, FN
    TP = ((df["prediction"] == 1) & (df["gt_label"] == 1)).sum()
    FP = ((df["prediction"] == 1) & (df["gt_label"] == 0)).sum()
    TN = ((df["prediction"] == 0) & (df["gt_label"] == 0)).sum()
    FN = ((df["prediction"] == 0) & (df["gt_label"] == 1)).sum()
    
    accu = (TP + TN)/(TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    
    evaluDic = {'TP-1-1':TP, 'FP-1-0':FP, 'TN-0-0':TN, 'FN-0-1':FN,
                'accu':round(accu, 5), 'precision':round(precision, 5), 
                'recall':round(recall, 5), 'f1_score':round(f1_score, 5)}
    
    return(evaluDic)

def ccg_indicators(ccg, smoothSigma=0.5, bin_size=1, ifVerbose=False, ifPlot=False, timebins=None,  figsize=(10,3), barColor='lightslategrey', spanColor='firebrick', ifSave=True, savePath=None, filename=None):
    
    '''
    peak height, peak index, peak lag, peak width (half max), peak to noise ratio are calculated using raw ccg
    entropy, temporal span are calculated using smoothed ccg
    '''
    
    # peak ------------------------------------------------
    center = len(ccg) // 2
    peak_height = np.max(ccg)
    peak_idx = np.argmax(ccg)
    peak_lag = (peak_idx - center) * bin_size

    # peak width (full width at half max) -----------------
    half_max = peak_height / 2
    left, right = peak_idx, peak_idx
    while left > 0 and ccg[left] > half_max:
        left -= 1
    while right < len(ccg) - 1 and ccg[right] > half_max:
        right += 1
    peak_width = (right - left) * bin_size

    # noise estimation (use tails) -----------------------
    tail_bins = np.r_[np.arange(25), np.arange(-25, 0)]
    noise_floor = np.mean(ccg[tail_bins])
    noise_std = np.std(ccg[tail_bins])
    peak_to_noise = (peak_height - noise_floor) / (noise_std + 1e-10)

    # gaussian smooth -------------------------------------------
    ccg_smooth = gaussian_filter1d(ccg, sigma=smoothSigma)

    # entropy -------------------------------------------
    ccg_prob = ccg_smooth / (np.sum(ccg_smooth) + 1e-10)
    ccg_entropy = entropy(ccg_prob, base=2)

    # temporal span above noise threshold ---------------
    thresh = noise_floor + 2 * noise_std
    left_span, right_span = center, center
    while left_span > 0 and ccg_smooth[left_span] > thresh:
        left_span -= 1
    while right_span < len(ccg_smooth) - 1 and ccg_smooth[right_span] > thresh:
        right_span += 1
    temporal_span = (right_span - left_span) * bin_size

    if ifVerbose:
        print('peak happens at bin', peak_idx, '=', peak_height, 'with time lag =', peak_lag )
        print('peak drop to half in', peak_width, 'bins')
        print('tails have mean =', round(noise_floor,3), 'with std =', round(noise_std,3), ' and peak_to_noise =', round(peak_to_noise,3))
        print('entropy =', round(ccg_entropy,3), 'bits. if in [0,2] - sharp/concentrated signal, ~4-6 bits - diffused, noisy, broad')
        print('from', left_span, 'to', right_span, 'bins, we have ccg above meanTail+2std('+str(int(thresh))+')' )


    if ifPlot:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].bar(timebins, ccg, color=barColor, width=1); ax[0].set_title('raw')
        ax[1].bar(timebins, ccg_smooth, color=barColor, width=1); ax[1].set_title('smoothed')
        ax[1].bar(timebins[left_span:right_span], ccg_smooth[left_span:right_span], 
                  color=spanColor, width=1, alpha=0.4, label='temporal span'); ax[1].legend(loc=1)
        plt.tight_layout()
        if ifSave:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(os.path.join(savePath, filename))
            plt.close()
        else:
            plt.show()
 

    
    return { "peak_height": peak_height,
            "peak_lag": peak_lag,
            "peak_halfMax_width": peak_width,
            "entropy": ccg_entropy,
            "peak_to_noise": peak_to_noise,
            "temporal_span": temporal_span}


# added 09/29/2025
def _ccg_event(pre_ms, post_ms, bin_size=1, duration=100, want_diffs=False):
    """
    Efficient CCG in ms without building the MxN diffs matrix.
    pre_ms, post_ms: 1D arrays of spike times in *milliseconds* (sorted or not).
    bin_size: bin width in ms (e.g., 1)
    duration: total window in ms (e.g., 100 -> +/-50 ms)
    want_diffs: if True, also returns the list of diffs within the window (for GLMCC)
    Returns: counts (n_bins,), centers (n_bins,), diffs_win (1D, optional)
    """
    pre = np.asarray(pre_ms, dtype=np.float64)
    post = np.asarray(post_ms, dtype=np.float64)
    pre.sort(); post.sort()

    half = duration / 2.0
    half_bins = int(round(duration / (2.0 * bin_size)))
    # edges and centers exactly like your original
    edges = np.arange(-half_bins - 0.5, half_bins + 1.5) * bin_size
    centers = (edges[:-1] + edges[1:]) / 2.0

    counts = np.zeros(edges.size - 1, dtype=np.int64)
    diffs_accum = [] if want_diffs else None

    # For each pre-spike, only look at post spikes in [t-half, t+half]
    for t in pre:
        lo = t - half
        hi = t + half
        i0 = np.searchsorted(post, lo, side='left')
        i1 = np.searchsorted(post, hi, side='right')
        if i1 > i0:
            diffs = post[i0:i1] - t  # in ms
            # accumulate histogram for this pre-spike
            counts += np.histogram(diffs, bins=edges)[0]
            if want_diffs:
                diffs_accum.append(diffs)

    if want_diffs:
        diffs_win = np.concatenate(diffs_accum) if diffs_accum else np.array([], dtype=np.float64)
        return counts, centers, diffs_win
    else:
        return counts, centers, None

def cal_efficientCCG(times, idx, duration, bin_size):
    """
    times: concatenated spike times (ms)
    idx:   same length as times, 0 for 'pre', 1 for 'post' (only two trains expected)
    duration, bin_size: in ms (e.g., 100, 1)
    """
    # Make sure units are ms for your pipeline:
    # (You already do s->ms conversion before calling)

    unique_idx = np.unique(idx)
    assert len(unique_idx) == 2, "This cal_CCG expects exactly two trains (pre=0, post=1)."
    id_pre, id_post = unique_idx[0], unique_idx[1]

    # Extract trains (keep float64 for precision)
    times1 = np.asarray(times[idx == id_pre], dtype=np.float64)   # pre
    times2 = np.asarray(times[idx == id_post], dtype=np.float64)  # post
    times1.sort(); times2.sort()

    # Cross-correlogram (pre -> post)
    cch, t, cch_diffs = _ccg_event(times1, times2, bin_size=bin_size, duration=duration, want_diffs=True)

    # Auto-correlograms using same routine; subtract zero-lag self-coincidences:
    ach1, tA, _ = _ccg_event(times1, times1, bin_size=bin_size, duration=duration, want_diffs=False)
    ach2, tB, _ = _ccg_event(times2, times2, bin_size=bin_size, duration=duration, want_diffs=False)
    # zero-lag is the center bin
    half_bins = int(round(duration / (2.0 * bin_size)))
    ach1[half_bins] = max(0, ach1[half_bins] - len(times1))
    ach2[half_bins] = max(0, ach2[half_bins] - len(times2))

    n_bins = 2 * half_bins + 1

    return (cch, cch_diffs, ach1, ach2, n_bins, half_bins, t)

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







# -------------------------------- Generate Dataloaders @09/30/2025--------------------------------#
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










# ------------------------ E-I balance analysis for Network Simualtion ------------------------#
def cal_global_EI_ratio(gEs, gIs, T, dt, eC='firebrick', iC='steelblue', ifPlot=False, ifSave=False, savePath=None, filename=None):

    '''
    global E-I balance for excitatory neurons in the network following Vogels et al. (2011)

    Arguments:
        - gEs: conductance levels of all incoming exc synapses; input shape should be: [Nt, N_neurons]
        - gIs: conductance levels of all incoming inh synapses
        - T: simulation time
        - dt
    '''
    range_t = np.arange(0, T, dt)
    mean_gE = np.mean(gEs, axis=1) 
    mean_gI = np.mean(gIs, axis=1)

    EI_ratio = mean_gI / (mean_gE + 1e-6) 


    if ifPlot:
        fig, ax = plt.subplots(3, 1, figsize=(20, 6))
        ax[0].plot(range_t, EI_ratio, c='r')
        ax[0].axhline(y=np.mean(EI_ratio), linestyle="--", color="black", label="Mean Ratio")
        ax[0].set_title("Mean gI/gE Over Time")

        ax[1].plot(mean_gE[3000:5000], c=eC, label='mean_gE')
        ax[1].plot(-1*mean_gI[3000:5000], c=iC, label='mean_gI'); ax[1].legend(); ax[1].set_title('it 3000-5000')

        ax[2].plot(mean_gE[-2000:], c=eC, label='mean_gE')
        ax[2].plot(-1*mean_gI[-2000:], c=iC, label='mean_gI'); ax[2].legend(); ax[2].set_title('it last 2000')

        plt.tight_layout()
        if ifSave:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(os.path.join(savePath, filename), bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    return EI_ratio

def cal_XCorr_EI_Current(Ecurr, Icurr, T, dt, eC='firebrick', iC='steelblue', ifPlot=False, ifSave=False, savePath=None, filename=None):

    '''
    global E-I balance for excitatory neurons in the network following Vogels et al. (2011)

    Arguments:
        - Ecurr: exc current input to this neuron; input shape should be: [Nt, N_neurons]
        - Icurr: inh current input to this neuron
        - T: simulation time
        - dt
    '''
    range_t = np.arange(0, T, dt)
    mean_Ecurr = np.mean(Ecurr, axis=1) 
    mean_Icurr = np.mean(Icurr, axis=1)
    norm_Ecurr = (mean_Ecurr - np.mean(mean_Ecurr)) / (np.std(mean_Ecurr) + 1e-6)
    norm_Icurr = (mean_Icurr - np.mean(mean_Icurr)) / (np.std(mean_Icurr) + 1e-6)

    # cross-correlation with time lag
    max_lag = 50 
    cross_corr = correlate(norm_Ecurr, norm_Icurr, mode="full")
    lags = np.arange(-max_lag, max_lag + 1) * dt 

    # only extrace the relevant part of the correlation
    center = len(cross_corr) // 2
    cross_corr = cross_corr[center - max_lag : center + max_lag + 1]

    EI_ratio = mean_Icurr/mean_Ecurr

    if ifPlot:
        fig, ax = plt.subplots(5, 1, figsize=(20, 5*2))
        ax[0].plot(lags, cross_corr, c='b')
        ax[0].axvline(x=0, linestyle="--", color="black", label="Zero Lag")
        ax[0].set_xlabel("Time Lag (ms)"); ax[0].set_ylabel("XCorr"); ax[0].set_title("Cross-Correlation of Excitation and Inhibition")

        ax[1].plot(mean_Ecurr, c=eC, label='Ecurr')
        ax[1].plot(mean_Icurr, c=iC, label='Icurr'); ax[1].legend(); ax[1].set_title("Mean Ecurr/ICurr Over Time")

        ax[2].plot(EI_ratio, c='green', label='Icurr/Ecurr'); ax[2].legend(); ax[2].set_title("Ratio ICurr/ECurr Over Time")
        
        ax[3].plot(mean_Ecurr[3000:5000], c=eC, label='mean_Ecurr')
        ax[3].plot(mean_Icurr[3000:5000], c=iC, label='mean_ICurr'); ax[3].legend(); ax[3].set_title('it 3000-5000')

        ax[4].plot(mean_Ecurr[-2000:], c=eC, label='mean_Ecurr')
        ax[4].plot(mean_Icurr[-2000:], c=iC, label='mean_ICurr'); ax[4].legend(); ax[4].set_title('it last 2000')

        plt.tight_layout()
        
        if ifSave:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(os.path.join(savePath, filename), bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    return lags, cross_corr







# ------------------------ save neuron parameters, neuron objs ------------------------#
def accumulated_sum(input_list):
    '''return accumulated sum of a list
    e.g., input_list = [50, 50, 50], 
    return [25, 75, 125]'''

    accu_sum = []
    for i in range(len(input_list)):
        half_accusum = input_list[i]/2 + np.sum(input_list[:i])
        accu_sum.append(half_accusum)

    return(accu_sum)

def dict_to_df(input_dict, columns=None, ifSave=False, savePath=None, filename=None):

    ''' dictionary to dataframe. Can be used to output neuron parameter space to dataframe'''
    df = pd.DataFrame.from_dict(input_dict, orient='index').reset_index()

    if type(columns) != type(None):
        if df.shape[1] != len(columns):
            raise ValueError('Input columns does not match dataframe number of columns')
        df.columns = columns
    
    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        df.to_excel(os.path.join(savePath, filename+'.xlsx'))

    return(df)

def extract_neuronParas_fromObjs(obj_list, keys=[],neuronNames=[], ifSave=False, savePath=None, filename=None):


    '''
    specifically save 'paras' in Neuron object to dataframe
    '''

    paras_df_list = []
    for obj in obj_list:
        dic = obj.__dict__['paras']
        df = pd.DataFrame.from_dict(dic, orient='index')
        paras_df_list.append(df)

    Paras_df = pd.concat(paras_df_list, axis=1).T
    if len(keys) != 0:
        Paras_df = Paras_df[keys]
    if len(neuronNames) != 0:
        Paras_df['neuron_name'] = neuronNames
        Paras_df = Paras_df.set_index('neuron_name', drop=True)
        

    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        Paras_df.to_csv(os.path.join(savePath, filename+'.csv'))

    return(Paras_df)

def neuronProperity_Report(T, dt, nType_list, nName_list, wE_list, wI_list, gEBar_list, gIBar_list, N_kwargs_list, ifSave=False, savePath=None, filename=None):
    
    '''
    generate a dataframe of neurons in the network, including:
        - gL ($gL = 1/R$)
        - input resistence
        - time constant
        - rheobase
        - F-I curve slope
        - gE/gI/WI/WE
        - EPSP/IPSP
    
    '''
    maxns=1000
    units = ['', 'nS', 'MΩ', 'ms', 'mV', 'mV', 'mV', 'nS', 'nS', '', '', 'nS', 'nS', 'mV', 'mV','', 'mV']
    
    
    propertyDF_list = []
    for neuronIdx in range(len(nType_list)):

        print('working on', nName_list[neuronIdx])

        neuron_type = nType_list[neuronIdx]
        if neuron_type==0: Ne = 1; Ni = 0; N = Ni + Ne
        else: Ne = 0; Ni = 1; N = Ni + Ne
            
    
        # Rheobase, slope
        N_kwargs = N_kwargs_list[neuronIdx] 
        current, FRs, slope, spikeStartPosition = F_I_curve(N_kwargs, T, dt, N, Ne, Ni, maxns, neuron_type,
                                                          startC=100, endC=300, numC=20, ifPlot=False)
        searchStart = current[spikeStartPosition-1]
        searchEnd = current[spikeStartPosition]
        Rheobase = find_Rheobase(N_kwargs, T, dt, N, Ne, Ni, maxns, searchStart, searchEnd, neuron_type)

        # other parameters
        wE = wE_list[neuronIdx]; wI = wI_list[neuronIdx]
        gE_bar = gEBar_list[neuronIdx]; gI_bar = gIBar_list[neuronIdx]
        N_obj = neuron.Neuron(T, dt, gE_bar, gI_bar, neuron_type, **N_kwargs)
        
        # summarize
        propertyDic = {'neuronName':nName_list[neuronIdx],
                       'g_Leak': N_obj.get('g_Leak'), 
                       'memResistance':N_obj.get('memResistance'),
                       'timeConstant':N_obj.get('tau_m'), 
                       
                       'V_rest/V_reset':N_obj.get('V_reset'), 
                       'V_excReversal':N_obj.get('V_excReversal'),
                       'V_inhReversal':N_obj.get('V_inhReversal'),
               
                       'gE_bar':gE_bar, 'gI_bar':gI_bar, 
                       'wE':wE, 'wI':wI, 
                       'gE':gE_bar*wE, 'gI':gI_bar*wI, 
                       'EPSP':-1/N_obj.get('g_Leak')*(gE_bar*wE*(N_obj.get('V_reset')-N_obj.get('V_excReversal'))),
                       'IPSP':-1/N_obj.get('g_Leak')*(gI_bar*wI*(N_obj.get('V_reset')-N_obj.get('V_inhReversal'))),
               
                       'F-I-Slope':slope, 'Rheobase':Rheobase}
        propertyDF = pd.DataFrame.from_dict(propertyDic, orient='index')
        propertyDF_list.append(propertyDF)
        
    
    allN_propertyDF = pd.concat(propertyDF_list, axis=1)
    allN_propertyDF['unit'] = units
    
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        allN_propertyDF.to_excel(os.path.join(savePath, filename+'.xlsx'))

        
    return(allN_propertyDF)

def save_data2Pickle(nameList, dataList, save_path, file_name):
    
    saveDic = {}
    
    if len(nameList) != len(dataList):
        raise ValueError("Input nameList and dataList must have same length")
    
    for i in range(len(nameList)):
        key = nameList[i]; data = dataList[i]
        saveDic[key] = data
        
    # save
    with open(os.path.join(save_path, file_name+'.pkl'), 'wb') as fp:
        pickle.dump(saveDic, fp)
        print(file_name+' pickle saved')

def generate_netSpk_report(T, netSpk, neuronName_list, neuronType_list, ifSave=False, savePath=None, filename=None):
    
    '''generate a dataframe to show neuron firings in the simulation'''
    
    spkTimes = netSpk[0]; spkTimes_list = list(spkTimes)
    spikeNeurons = netSpk[1]; spikeNeurons_list = list(spikeNeurons)
    uniqueNeuronIdx = [int(i) for i in set(spikeNeurons)]

    # how many times each neuron fire
    occurrence = {int(item): spikeNeurons_list.count(item) for item in uniqueNeuronIdx}
    # what are the names/types of neuron
    spk_neuronNames = np.array(neuronName_list)[uniqueNeuronIdx] #[neuronName_list[i] for i in occurrence.keys()]
    spk_neuronTypes = np.array(neuronType_list)[uniqueNeuronIdx]
    # create a base dataframe
    spkNeuron_Summary = pd.DataFrame.from_dict(occurrence, orient='index', columns=['firing_times'])
    spkNeuron_Summary['neuron_name'] = spk_neuronNames
    spkNeuron_Summary['neuron_type'] = spk_neuronTypes
    spkNeuron_Summary['firing_rate'] = spkNeuron_Summary['firing_times']/(T/1000)

    # spk times of each neuron
    neuornspk_times = {}
    for nid in uniqueNeuronIdx:
        neuornspk_times[nid] = spkTimes[np.where(spikeNeurons==nid)]
    neuornspk_time_df = pd.DataFrame.from_dict(neuornspk_times, orient='index')

    # concat
    netSpk_Summary = pd.concat([spkNeuron_Summary, neuornspk_time_df], axis=1)
    netSpk_Summary = netSpk_Summary.sort_values('firing_times', ascending=False)

    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        netSpk_Summary.to_excel(os.path.join(savePath, filename+'.xlsx'))


    return(netSpk_Summary)
    
def ORN_PN_firingSummary(neuronObj_list, T, dt, num_ORNs, num_PNs, ifEvaluate_FRExp=True, ifVerbose=True, ifSave=False, savePath=None, filename=None):
    
    '''
    get ORN actual firing rate, PN firing rate
    '''
    ORN_actualFRs = get_firingRates(neuronObj_list, list(range(num_ORNs)), T, dt, spkKey='assignSpkTrain')
    PN_FRs = get_firingRates(neuronObj_list, list(range(num_ORNs, num_ORNs+num_PNs)), T, dt, spkKey='spkTrain')

    
    if ifEvaluate_FRExp:
        evaluate_FR_exp(ORN_actualFRs, histTitle='ORN', thresholdPer=0.2, ifPlot=True, ifVerbose=ifVerbose)
        evaluate_FR_exp(PN_FRs, histTitle='PN', thresholdPer=0.2, ifPlot=True, ifVerbose=ifVerbose)
        
    
    # generate PN/ORN firing ranking summary
    ORN_FR_Rank = pd.DataFrame.from_dict({'ORN_input_firing_rate':ORN_actualFRs, 
                                      'ORN_neuorn/obj_id':range(num_ORNs)}
                                    ).sort_values('ORN_input_firing_rate', ascending=False)
    PN_FR_Rank = pd.DataFrame.from_dict({'PN_firing_rate':PN_FRs, 
                                         'PN_neuorn_id':range(num_PNs),
                                         'PN_Obj_id':range(num_ORNs, num_ORNs+num_PNs)}
                                       ).sort_values('PN_firing_rate', ascending=False)
    ORN_PN_FR_Rank = pd.concat([ORN_FR_Rank, PN_FR_Rank], axis=1).reset_index(drop=True)
    
    
    if ifVerbose:
        print('ORN top 5 firing neurons:', ORN_PN_FR_Rank['ORN_neuorn/obj_id'][0:5].values)
        print('PN top 5 firing enurons:', ORN_PN_FR_Rank['PN_Obj_id'][0:5].values)
    
    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        ORN_PN_FR_Rank.to_excel(os.path.join(savePath, filename+'.xlsx'))


    return(ORN_PN_FR_Rank, ORN_actualFRs, PN_FRs)  
    
def savesimulatorObjResult(simulatorObj, neuronObj_list, ORNs_FRs, num_ORNs, num_PNs, neuronName_list, neuronType_list, 
                           descriptiveWords, savePath, ifEvaluate_FRExp=True, netSpk_Summary=None, ORN_PN_FR_Rank=None):
    
    
    '''
    specifically write for fruit fly olfactory simualtion result
    '''
    
    T = simulatorObj.T; dt = simulatorObj.dt; N = simulatorObj.N

    if type(netSpk_Summary) == type(None):
        netSpk_Summary = generate_netSpk_report(T, simulatorObj.get('netSpk'), neuronName_list, neuronType_list,
                                            ifSave=True, savePath=savePath, filename='netSpkSummary')
    if type(ORN_PN_FR_Rank) == type(None):
        ORN_PN_FR_Rank,ORN_actualFRs,PN_FRs = ORN_PN_firingSummary(neuronObj_list, T, dt, num_ORNs, num_PNs, ifEvaluate_FRExp=ifEvaluate_FRExp, 
                                              ifVerbose=False,ifSave=True, savePath=savePath, filename='ORN_PN_FR_Rank')
    
    
    
    simuResult_dict = {}

    # general
    simuResult_dict['T'] = T
    simuResult_dict['dt'] = dt
    simuResult_dict['N'] = N
    simuResult_dict['Ne'] = simulatorObj.get('Ne')
    simuResult_dict['Ni'] = simulatorObj.get('Ni')
    simuResult_dict['ns'] = simulatorObj.get('ns')
    simuResult_dict['range_t'] = neuronObj_list[0].get('range_t')


    # about ORN
    simuResult_dict['ORNs_FRs'] = ORNs_FRs
    simuResult_dict['ORN_actualFRs'] = ORN_actualFRs
    simuResult_dict['ORNs_spkTrains'] = get_keyValues_2_2DArray(neuronObj_list, list(range(num_ORNs)), 
                                                                T, dt, Key='assignSpkTrain')

    # about PN
    PN_neuronIdx = list(range(num_ORNs, num_ORNs+num_PNs))
    simuResult_dict['PNs_para'] = dict_to_df(neuronObj_list[num_ORNs+1].get('paras'), columns=['PN Parameter', 'Values'])
    simuResult_dict['PNs_FRs'] = PN_FRs
    simuResult_dict['PNs_spkTrains'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='spkTrain')
    simuResult_dict['PNs_memPotential'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='memPotential')
    simuResult_dict['PNs_synTrace'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='synTrace')
    simuResult_dict['PNs_gE'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='gE')
    simuResult_dict['PNs_gI'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='gI')
    simuResult_dict['PNs_EPSP'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='EPSP')
    simuResult_dict['PNs_IPSP'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='IPSP')
    simuResult_dict['PNs_Xcurr'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='Xcurr')
    simuResult_dict['PNs_LeakC'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='LeakC')
    simuResult_dict['PNs_spkTimes'] = get_keyValues_2_2DArray(neuronObj_list, PN_neuronIdx, T, dt, Key='spkTimes')

    
    # about LIN
    LIN_neuronIdx = list(range(num_ORNs+num_PNs, N))
    simuResult_dict['LINs_para'] = dict_to_df(neuronObj_list[num_ORNs+num_PNs+1].get('paras'), columns=['LIN Parameter', 'Values'])
    simuResult_dict['LINs_FRs'] = get_firingRates(neuronObj_list, LIN_neuronIdx, T, dt, spkKey='spkTrain')
    simuResult_dict['LINs_spkTrains'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='spkTrain')
    simuResult_dict['LINs_memPotential'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='memPotential')
    simuResult_dict['LINs_synTrace'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='synTrace')
    simuResult_dict['LINs_gE'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='gE')
    simuResult_dict['LINs_gI'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='gI')
    simuResult_dict['LINs_EPSP'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='EPSP')
    simuResult_dict['LINs_IPSP'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='IPSP')
    simuResult_dict['LINs_Xcurr'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='Xcurr')
    simuResult_dict['LINs_LeakC'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='LeakC')
    simuResult_dict['LINs_spkTimes'] = get_keyValues_2_2DArray(neuronObj_list, LIN_neuronIdx, T, dt, Key='spkTimes')


    # about connectivity
    simuResult_dict['CMWParas'] = dict_to_df(simulatorObj.get('CMWParas'), columns=['Connectivity', 'Values'])
    simuResult_dict['CM'] = simulatorObj.get('CM')
    simuResult_dict['CMW'] = simulatorObj.get('CMW')
    simuResult_dict['CMWs'] = simulatorObj.get('CMWs')
    simuResult_dict['outgoingCs'] = simulatorObj.get('outgoingCs')
    simuResult_dict['incomingCs'] = simulatorObj.get('incomingCs')

    
    # about network summary
    simuResult_dict['netSpk'] = simulatorObj.get('netSpk')
    simuResult_dict['netSpk_Summary'] = netSpk_Summary
    simuResult_dict['ORN_PN_FR_Rank'] = ORN_PN_FR_Rank


    # a note
    simuResult_dict['notes'] = descriptiveWords

    
    # save
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    with open(os.path.join(savePath, 'simuResult.pkl'), 'wb') as fp:
        pickle.dump(simuResult_dict, fp)
        print('saved successfully')
        
def timeSeries_stationary_test(spike_train):

    adf_result = adfuller(spike_train)
    print(f"ADF Statistic: {adf_result[0]} with p-value = {adf_result[1]}. And critical values at all levels are {adf_result[4]}" )

    if adf_result[1] < 0.05:
        return True
    else:
        return False

def cal_dynamicTimeWarping_plot(df, col1, col2, figsize=(4, 4), col1_styple='b-', col2_styple='b-', ifPlot=False, ifSave=False, savePath=None, filename=None):

    '''
    perform dynamic time warping, plot path and return distance

    '''
    
    s1 = df[col1].to_numpy().reshape((-1, 1))
    s2 = df[col2].to_numpy().reshape((-1, 1))
    path, sim = metrics.dtw_path(s1, s2)

    # plot
    if ifPlot:
        plt.figure(1, figsize=figsize)
        left, bottom = 0.01, 0.1; w_ts = h_ts = 0.05; left_h = left + w_ts + 0.02; width = height = 0.6; bottom_h = bottom + height + 0.02
        rect_s_y = [left, bottom, w_ts, height]; rect_gram = [left_h, bottom, width, height]; rect_s_x = [left_h, bottom_h, width, h_ts]
        ax_gram = plt.axes(rect_gram); ax_s_x = plt.axes(rect_s_x); ax_s_y = plt.axes(rect_s_y)
        
        mat = cdist(s1, s2)
        ax_gram.imshow(mat, origin='lower')
        ax_gram.axis("off"); ax_gram.autoscale(False)
        ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-", linewidth=3.)
        
        ax_s_x.plot(np.arange(s2.shape[0]), s2, col2_styple, linewidth=3.)
        ax_s_x.axis("off")
        
        ax_s_y.plot(- s1, np.arange(s1.shape[0]), col1_styple, linewidth=3.)
        ax_s_y.axis("off")
        
        plt.title('distance = '+str(round(sim,4)),  x=8, y=1.2); plt.tight_layout()
        
        # save/show
        if ifSave:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(os.path.join(savePath, filename))
            plt.close()
        else:
            plt.show()

    return(sim)








