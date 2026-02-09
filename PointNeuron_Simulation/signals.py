'''
# signals.py
# Author: Xiaoqian Sun, 06/04/2024
# Generate signals, such as: constant/sine-like/regualr-pulse/poission spikes input
# These signals can be external/background inputs to neurons, or can be spike trains attached to neurons
'''


# Import Packages
#========================================================================================
import os
import math
import numpy as np
import scipy as sp
from collections import Counter
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')



# Functions
#========================================================================================
def ramp_input(Nt, t_ramp, amp=22, ifPlot=False):
    
    '''
    create exponential ramping input in t_ramp and flat (near 0) input in t_flat=T-t_Ramp
    note that Nt = T/dt
    E.g.,
        Nt=1000 
        ramp = ramp_input(Nt, int(Nt/2))
        will give you a exp ramping from 0-22 in first 500 steps, and all 0 in the second 500 steps
    '''
    
    
    t_flat = int(Nt-t_ramp)
    ramp = np.linspace(-t_ramp, t_flat, Nt)

    scale1 = t_ramp/3         # let ramp[0] starts from np.exp(-3)
    scale2 = ramp[t_ramp]/50  # let ramp[t_ramp] goes down to np.exp(-50)
    ramp[:t_ramp] = np.exp(ramp[:t_ramp]/scale1) # first half input, 0-1(near)
    ramp[t_ramp:] = np.exp(-ramp[t_ramp:]/scale2)  # second half input, 1-0(near)
    ramp = ramp*amp
    
    if ifPlot:
        plt.figure(figsize=(20, 2))
        plt.plot(ramp)
        plt.title('ramping stops at'+str(t_ramp)); plt.show()
        
        
    return (ramp)

def sin_input(Nt, dt, num_periods=4, amp=20, ifPlot=False):
    
    
    '''
    a sine-like intput
    E.g.,
        Nt=1000, dt=0.1
        sin = sin_input(Nt, dt)
        will give you 4 sins 1000 steps, max=20, min=0, start=10
    '''
    
    if not num_periods:
        num_periods = int(1/dt)
    
    x = np.linspace(-num_periods*np.pi, num_periods*np.pi, Nt)
    sin = np.sin(x)

    sin01 = (sin-np.min(sin))/(np.max(sin)-np.min(sin))
    sin01 = sin01*amp

    
    if ifPlot:
        plt.figure(figsize=(20, 2))
        plt.plot(sin01)
        plt.title('sin (0-1) input'); plt.show()
    
    
    return(sin01)

def pulse_input(Nt, step=50, last=1, amp=20, ifStart0=True, ifPlot=False):
    
    '''
    create a pulse input, each session has # step 0 + # last 1
    e.g., with step=5, last=1 [0 0 0 0 0 1 0 0 0 0 0 1...] (5 0s + 1 + 5 0s + 1 ...)
    e.g., with step=5, last=2 [0 0 0 0 0 1 1 0 0 0 0 0 1 1 ...] (5 0s + 1 1 + 5 0s + 1 1 ...)
    E.g.,
        Nt=1000, dt=0.1
        pulse = pulse_input(Nt, step=50, last=5, ifStart0=True)
        will give you a series of regular pulses, max=20, min=0, start=0, 18 pulses in total

    '''
    numPulse = Nt//(step+last)
    
    pulse = []
    for i in range(1, numPulse+10):
        pulse += [0]*step + [1]*last
    
    pulse = np.array(pulse)*amp
    pulse = pulse[:Nt]
    
    if ifPlot:
        plt.figure(figsize=(20, 2))
        plt.plot(pulse)
        plt.title('pulse input'); plt.show()
        
    return(pulse)

def noisyOU_input(Nt, dt=0.1, tau_noise=10, sigma_noise=10, mu_noise=180, ifPlot=False, figsize=(10, 2), c='b', ifSave=False, savePath=None, filename=None):
    
    '''
    eANN noisy input
    follow method mentioned in: https://neuronaldynamics.epfl.ch/online/Ch8.S1.html
    see `Desktop/InferConnectivity/A_Archive/GLMCC/1-explore_OU_Ib.ipynb` for tech details
    Arguments:
        - tau_noise: how fast the noise decays to zero and how smooth or rapid the fluctuations are
            - small: noise fluctuates rapidly -  fast, jittery noise
            - large: noise changes slowly - slowly fluctuating input
            - fast, irregular spiking → use small tau_noise
            - slow variability in firing rate → Use large tau_noise
        - sigma_noise: fluctuation around the mu_noise
            - note that sigma_noise got muptiplied to xi (samples from "standard normal distribution" [-3, 3])
            - if sigma_noise = 1, then I_b mostly mu_noise-3, mu_noise+3

    
    '''
    
    # white noise
    xi = np.random.randn(Nt)

    I_noise = np.zeros(Nt)
    for t in range(1, Nt):
        I_noise[t] = I_noise[t-1] + dt*( -I_noise[t-1]/tau_noise) + sigma_noise * xi[t] * np.sqrt(dt)
    I_noise += mu_noise
    
    
    if ifPlot:
        plt.figure(figsize=figsize)
        plt.plot(np.arange(Nt) * dt, I_noise, color=c)
        plt.xlabel("Time (ms)"); plt.ylabel("Noise Current (pA)")
        plt.title("Ornstein-Uhlenbeck Noise Process")
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

       
    return I_noise

def PoissonSpkTrain(dt, range_t, rate, n, myseed=False):
    """
    Aim:
        Generates poisson trains for a bunch of neurons
    How:
        generate a matrix with shape (num_neuron, num_t_points). 
        threshold=rate * (dt / 1000.), e.g., if rate=10, then threshold=0.01
        any value in the matrix > threshold will be a spike (value 1), and other places will be value 0

    Args:
        - dt: parameter dictionary, [ms]. Note that later, dt = dt/1000 = 0.0001s
        - rate: noise amplitute [Hz]
        - n: number of Poisson trains (number of neurons)
        - myseed: random seed. int or boolean

    Returns:
        pre_spike_train : spike train matrix, ith row represents whether
                          there is a spike in ith spike train over time (1 if spike, 0 otherwise)
    """
    

    # set random seed
    if myseed:
        np.random.seed(seed=myseed)
    else:
        np.random.seed()

    # number of timepoints 
    Lt = range_t.size

    # if one sinlge num for rate, convert to array (just to accomodate different input of rate, an array or a num)
    if type(rate) in [int, float, np.float64, np.int64]: rate = np.array([rate]) 

    # for each time step, generate a random number uniformly distributed between 0 and 1
    # this random number represents the probability of a spike occurring during that time step
    u_rand = np.random.rand(n, Lt)

    # probability of observing a spike for a small timestep
    # e.g., rate=10 (spikes/second); T=100ms, dt=0.01
    # then probability of a spike in timebin=0.01ms is r/1000 * dt = 0.0001
    # that means during each timestep, 0.01% chance a spike will occur
    P_spk_timestep = rate*(dt/1000)

    # generate Poisson train
    # if the random number is < P_spk_timestep, there's a spike
    # this is Poisson distribution approximate using Binomial distribution 
    poisson_train = 1 * (u_rand < P_spk_timestep[:, np.newaxis] ) # type-int

    if n==1:
        return poisson_train[0]
    else:
        return poisson_train

def modify_absRefractory(spkTrain, Ntref):

    '''
    Follow the method metioned in dcCCH paper 
    The presynaptic neuron exhibited Poisson spiking modified by a 2 ms refractory period. 
    That is: any spike occurring withint 2ms (Ntref=tref/dt) of the previous spike should be removed
    See tech details in `InferConnectivity - 0-spkTrain_burstingFactor.ipynb`
    '''
    
    modifiedARP_spkTrain = np.copy(spkTrain)
    previousSpk_idx = -1
    for i, spike in enumerate(spkTrain):
        if spike == 1:
            if i - previousSpk_idx < Ntref:
                modifiedARP_spkTrain[i] = 0
            else:
                previousSpk_idx = i


    return modifiedARP_spkTrain

def add_burstActivity(spkTrain, dt, BP, delayWindow=(2, 7)):

    '''
    Add bursting activity modulated using a geometric distribution
        - each spike has a probability $p$ of terminating the bust (being the last spike) - we set this probability
        - If a spike initiated a burst, subsequent spikes are added probabilistically, which each new spike continuing the burst with $(1-p)$
        - while np.random.rand() > p, we add a new spike
            - larger p - shorter burst - $p=0.5$ introduces 2 spikes in burst - double the spike #
            - smaller p - longer burst - $p=0.2$ leads to 5 spikes in burst 
    See tech details in `InferConnectivity - 0-0-burstingGeometric_ARP.ipynb`
    '''

    nSamples = len(spkTrain)

    bursting_spkTrain = np.copy(spkTrain)
    for i, spike in enumerate(spkTrain):
        
        if spike == 1:                                # if we see a spike
            isi = np.random.uniform(*delayWindow)     # random delay to add another spike
            while np.random.rand() > BP:              # chance to add another spike
                
                delay_SpkIdx = i + int(isi/dt)
                if delay_SpkIdx < nSamples:
                    bursting_spkTrain[delay_SpkIdx] = 1
    
                isi = np.random.uniform(*delayWindow)  # next ISI randomly


    return (bursting_spkTrain)


def exp_FRs(mean_FR, num_neuron, ifPlot=False):
    '''
    given one mean firing rate, sample an array of firing rates from exponential distribuiton
    So that the hitogram of firing rates is exponential shape
    '''
    firing_rates = np.random.exponential(scale=mean_FR, size=num_neuron).astype(int)
    
    if ifPlot:
        plt.figure(figsize=(5, 3))
        plt.hist(firing_rates)
        plt.title('Probability distribution of ORN firing rates'); plt.show()

    return(firing_rates)

def adjust_expSpkTrain(spkTrain, aim_firingRate, T, dt, range_t):
    '''
    the spkTrain generated from Poisson process might not have actual firing rate of the aim_firingRate
    which will make the actual firing rate distribution not an exp shape
    so, iteratively adjust spkTrain to make the actual firing rate match desired firing rate
    which wil make actual firing rates distribution more like an exp
    '''
    
    actual_fr = spkTrain.sum()/(T/1000)
    
    if actual_fr < aim_firingRate:
        
        while actual_fr < aim_firingRate:
            makeup_fr = aim_firingRate - actual_fr
            spkTrain |= PoissonSpkTrain(dt, range_t, makeup_fr, 1, myseed=False)
            actual_fr = spkTrain.sum()/(T/1000)
    
    elif actual_fr > aim_firingRate:
        
        while actual_fr > aim_firingRate:
            spkLocs = np.where(spkTrain)[0]
            if len(spkLocs) > 0:
                spkTrain[np.random.choice(spkLocs)] = 0
            actual_fr = spkTrain.sum()/(T/1000)
            
            
    return(spkTrain)

def generate_synapticTrace(pre_spike_train, Lt, dt, tau_stdp):
    """

    Aim:
      track of presynaptic spikes (both inh/exc)
      the P is like how the neurontransmitter concentrate would change when got released 
      to cleft after a presynaptic spike happens

    Arguments:
      - pre_spike_train_ex: binary spike train input from presynaptic neuron

    Returns:
        synaptic trace x_j, 𝜏_𝑆𝑇𝐷𝑃(𝑑𝑥_𝑖)/𝑑𝑡=−𝑥_𝑖 , and increase with each spike 𝑥_𝑖→𝑥_𝑖+1


    """

    if len(pre_spike_train.shape) == 1:
        P = np.zeros(len(pre_spike_train))
        for it in range(Lt - 1):
            dP = - (dt/tau_stdp)*P[it] + 1*pre_spike_train[it+int(dt)]
            P[it + 1] = P[it] + dP
    else:
        P = np.zeros(pre_spike_train.shape)
    
        for it in range(Lt - 1):
            dP = - (dt/tau_stdp)*P[:, it] + 1*pre_spike_train[:, it+int(dt)]
            P[:, it + 1] = P[:, it] + dP

    return P

def square_kernel(size, ifPlotKernel=False):
    '''
    create a normalized square kernel (sum of the kernel = 1)
    '''
    
    square_kernel = np.ones(size)  
    norm_squareKernel = square_kernel / square_kernel.sum()
    
    if ifPlotKernel:
        plt.figure(figsize=(20, 2))
    plt.plot(norm_squareKernel, 'b-', linewidth=2, label='Square Kernel')
    plt.grid(True); plt.legend(); plt.show()
    
    return(norm_squareKernel)
    
def gaussian_kernel(size, sigma=10, ifPlotKernel=False):
    
    ''' 
    generate a normalized Gaussian kernel (area under the curve =1)
    this function yields the same result as using: 
        - kernel=sp.signal.gaussian(2*size+1, std=sigma); kernel = kernel/np.sum(kernel)
    '''
    
    size = int(size) // 2
    x = np.linspace(-size, size, 2*size+1)
    kernel = np.exp(-(x**2)/(2 * sigma**2))
    normed_kernel = kernel/np.sum(kernel)
    
    if ifPlotKernel:
        plt.figure(figsize=(20, 2))
        plt.plot(normed_kernel, 'b-', linewidth=2, label='Gaussian Kernel')
        plt.grid(True); plt.legend(); plt.show()
    
    return (normed_kernel)

def binning_spkTrain(spike_train, N, dt, simuT, bin_dt, binType='binary', ifVerbose=False):
    '''
    binning spike train, sum number of spikes in a bin size window, and use that as new spike sequence
    for most simulation data, binType doesn't matter since we barely see several spikes in bin_dt ms
    Note that spike_train shoud be in shape [num_neuron, num_timesteps]
    '''
    
    ori_sL = int(simuT/dt)
    binSize = int(bin_dt/dt)
    binnedLength = int(ori_sL/binSize)
    
    # binning
    if binSize==1:
        spkTrain_binned = spike_train
    elif binSize<1:
        raise Exception('binSize needs to be larger than 1')
    else:
        spkTrain_binned = np.zeros((N, binnedLength))
        bins = range(0, ori_sL, binSize)
        for i in range(N):
            spks_in_bin = np.array([np.sum(spike_train[i,:][bin:bin+binSize]) for bin in bins ])
            
            # no matter how many spikes in one bin, we use 1 to mark it
            # so, 1 means at least 1 spike in that bin
            # else, we use the actual # of spikes in each bin
            if binType == 'binary':
                spks_in_bin[spks_in_bin>=1]=1
                
            spkTrain_binned[i,:] = spks_in_bin[0:binnedLength]

    if ifVerbose :
        print('Originally, each neuron trace has length',ori_sL)
        print('After binning, spike train of each neuron is represent by a binary sequenc with length', binnedLength)
        print('To bin, we sum # of spikes every', binSize, 'steps')
        print('spkTrain_binned.shape after binning:', spkTrain_binned.shape, '| original spike_train.shape:', spike_train.shape)

        
    return (spkTrain_binned)

def convolve_spkKernel(spike_train, kernelSize=20,kernelType='Gaussian',sigma=20, ifPlotKernel=False, mode='same',method='direct', ifPlotLA=False):
    
    '''
    Aim:
        In Renart_2010, Local activity of cell 𝑖 at time 𝑡 with resolution 𝑇 
        is a convolution of 𝑠_𝑖 (𝑡)  with a normalized kernel 𝐾_𝑇 (𝑡)
        check E1 in 1_1_SpikeCountCorrelationCoefficient.ipynb
        
    Arguments:
        - spike_train: an array with binary values, 0-no spikes, 1-spikes
        - kernel can be a normalized square kernel or Gaussian Kernel
            - kernelSize: length of the kernel, ranging from 20ms to 50ms
            - kernelType: Gaussian or square
            - sigma: needed for Gaussian kernel
        - mode: choose 'same' to make sure the output has the same lenght with spike_train
        - method: direct and fft would yeild almost same result but direct runs faster
    '''
    
    # kernel
    if kernelType == 'Gaussian':
        kernel = gaussian_kernel(kernelSize, sigma=sigma, ifPlotKernel=ifPlotKernel)
    elif kernelType == 'Square':
        kernel = square_kernel(kernelSize, ifPlotKernel=ifPlotKernel)
    
    
    # convolve spike train with the kernel
    local_activity = sp.signal.convolve(spike_train, kernel, mode=mode, method=method)
    
    # plot
    LA_max = round(max(local_activity), 2)
    if ifPlotLA:
        plt.figure(figsize=(20, 2))
        
        t_sp = np.where(spike_train>0.8)[0]
        plt.plot(t_sp, np.ones(len(t_sp))*LA_max, '|', ms=20, markeredgewidth=2, color='k')
        plt.plot(local_activity, 'b-', linewidth=2, label='local_activity')
        plt.grid(True); plt.legend(); plt.show()
    
    
    return (local_activity)

def cal_local_activity(N,spike_train,kernelSize=20,kernelType='Gaussian',sigma=10,ifPlotKernel=False,mode='same',method='direct',ifPlotLA=False):
    '''
    simply apply function convolve_spkKernel() on each neuron's spike train
    check E1 in 1_1_SpikeCountCorrelationCoefficient.ipynb
    Note that spike_train shoud be in shape [num_neuron, num_timesteps]
    '''
    
    
    local_activity = np.zeros(spike_train.shape) 
    
    for i in range(N):
        local_activity[i, :] = convolve_spkKernel(spike_train[i,:],kernelSize=kernelSize,kernelType=kernelType,sigma=sigma, 
                       ifPlotKernel=ifPlotKernel, mode=mode,method=method, ifPlotLA=ifPlotLA)
    
    return (local_activity)


# below 2 functions are for transfer entropy
def lagged_series(spkArray, lag):
    '''
    note that this is getting full history of a time series
    or put it in this way, it getting all pieces with lag window size

    This Function is for Transfer Entropy Method
    
    '''
    history = np.array([spkArray[i - lag:i] for i in range(lag, len(spkArray))])

    return history

def TrasnferEntropy_2Neurons(spkTrain_A_source, spkTrain_B_target, lag=5):

    '''
    The goal is to determine how much additional information the past of neuron 𝐴 (𝑎_𝑡^((𝑙) ))​ provides about 
    the future state of neuron 𝐵 (𝑏_(𝑡+1)), beyond what is provided by neuron B’s own past (𝑏_𝑡^((𝑘)))
    
    A potential causal influence or information flow from neuron 𝐴 to neuron 𝐵.

    Set 
        - furture state of B is b_t+1
        - history state of B is bk, history state of A is al, where k and l are lag window (how many timesteps of b/a we look at)
    
    Formula:
        TE = sum of P(b_t+1, bk, al) * log of P(b_t+1 | bk, al)/P(b_t+1, bk)

    Steps:
        - create lagged spkTrain_A and spkTrain_B
        - create furture state of spkTrain_B

        - caluclate joint P(b_t+1, bk, al)
        - calculate margin P(bk, al), P(b_t+1, bk)
        - calculate conditional P(b_t+1 | bk, al)
        - calculcate TE
    '''

    
    lagged_spkTrain_A = lagged_series(spkTrain_A_source, lag)
    lagged_spkTrain_B = lagged_series(spkTrain_B_target, lag)
    future_B = spkTrain_B_target[lag:]

    
    total = len(lagged_spkTrain_B)
    
    counter_join = Counter()
    counter_margin_ba = Counter()
    counter_margin_b1b = Counter()
    for b_next, b_hist, a_hist in zip (future_B, lagged_spkTrain_B, lagged_spkTrain_A):
        joint_event = (tuple(b_hist), tuple(a_hist), b_next)
        margin_event_ba = (tuple(b_hist), tuple(a_hist))
        margin_event_b1b = (tuple(b_hist), b_next)

        counter_join[joint_event] +=1
        counter_margin_ba[margin_event_ba] +=1
        counter_margin_b1b[margin_event_b1b] +=1

    p_join_dic = {key: value/total for key, value in counter_join.items()}
    p_margin_ba_dic = {key: value/total for key, value in counter_margin_ba.items()}
    p_margin_b1b_dic = {key: value/total for key, value in counter_margin_b1b.items()}

    # conditional p
    cond_prob_dic = {}
    for key, join_p in p_join_dic.items():
        b_hist, a_hist, b_next = key
        if (b_hist, a_hist) in p_margin_ba_dic:
            cond_prob_dic[key] = join_p/p_margin_ba_dic[(b_hist, a_hist)]


    # calculate TE
    te_sum = 0

    for b_next, b_hist, a_hist in zip (future_B, lagged_spkTrain_B, lagged_spkTrain_A):
        event = (tuple(b_hist), tuple(a_hist), b_next)
    
        if event in p_join_dic and (tuple(b_hist), tuple(a_hist)) in p_margin_ba_dic:
            p_joint = p_join_dic[event]
            p_cond_b_a = cond_prob_dic[event]
            p_cond_b1b = p_margin_b1b_dic[tuple(b_hist), b_next]
    
            if p_cond_b_a>0 and p_cond_b1b>0:
                te_sum += p_joint*math.log(p_cond_b_a/p_cond_b1b)

    return(te_sum)







