# Functions.py
#
# Author: Xiaoqian Sun, 06/2023 
#
# Essential functions used in neuronal model simulation




# Import Packages
#========================================================================================
import os 
# import cv2
import time
import h5py
import math
import random
import pickle
import imageio
# import rasterio
import numpy as np
import pandas as pd
from tifffile import imwrite as imsave
# from rasterio.plot import show
import matplotlib.pyplot as plt
from oasis.functions import deconvolve

from scipy import optimize
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid

import warnings
warnings.filterwarnings('ignore')





# Functions
#========================================================================================
def PoissonSpikeTrain(rate, dur, dt=0.0001):
    
    '''
    Aim:
        generate poisson spike train with firing rate and duration using exponential distribution
        Note that the generate spike times have randomness in there 
        
    Arguments:
        rate: spikeRate, defined ahead
        dur: duration of whole signal
        dt: delta t
    '''
    
    spikeTimes = []
    for t in np.arange(0, dur, dt):
        if rate*dt > random.random():
            spikeTimes.append(t)
            
    
    return(spikeTimes)
            

def Calcium2Fluor(ca, ca_rest=50, kd=250, dffmax=93):
    '''
    Aim:
        conversion function, teransform calcium concentration to DFF value
        This function is called when ca_genmode='satDFF' in modelCalcium() when calculating snr
        This function is called in SingleFluorTransient() in such a way:
           lowdff = Calcium2Fluor(lowtmp,ca_p.ca_rest,exp_p.kd, exp_p.dffmax);
        
    Argument:
        - ca: ca peak amplitude, calculated by adding ca_amp+ca_rest. Since 'satDFF', saturation effect should be considered.
              ca_amp1 = ca_amp/(1+ca_kappas+kappab)
              ca_tau1 = (1+ca_kappas+kappab)/ca_gamma
              PeakCa = ca_amp1 + ca_rest
              ca = PeakCa
        - ca_rest: ca resting level (nm)
        - kd: dye dissociation constant
        - dffmax: saturating dff max in %
        
    '''
    
    fout = dffmax*(ca-ca_rest)/(ca+kd)
    
    return(fout)    
    

def spkTimes2Calcium(spkT,tauOn=0.01,ampFast=0.08,tauFast=0.06,ampSlow=0.03,tauSlow=0.8,frameRate=1000,duration=30):
    
    '''
    Aim:
        This funciton is to translate one spike to calcium trace using exponential function. 
        this function is called around line 96 in modelCalcium.m
    Note:
        the default values for tauOn, ampFast, tauFast, ampSlow, tauSlow are for OGB1-AM indicator
        % tauOn ... 10ms; ampFast ... 8%; tauFast ... 60ms; ampSlow ... 3%; tauSlow ... 800ms
    '''
    
    x = np.arange(0, duration, 1/frameRate)
    y = (1-(np.exp(-(x-spkT)/tauOn)))*(ampFast*np.exp(-(x-spkT)/tauFast))+(ampSlow*np.exp(-(x-spkT)/tauSlow))
    
    
    y[x<spkT] = 0
    y[np.isnan(y)] = 0
    
    return y
    

def spkTimes2FreeCalcium(spkT,Ca_amp=7600,Ca_gamma=400,Ca_onsettau=0.02,Ca_rest=50,
                         kappaS=100,Kd=250,Conc=50000,frameRate=1000,duration=30,ifPlot=False):
    
    '''
    Aim: 
        returns modeled free calcium trace derived from list of spike times (spkT)
        calculated based on buffered increment Ca_amp and taking indicator binding into account
        this function is called in modelCalcium.m, Peeling()
        
    Note:
        - for duration, should use 30, not 29.99997. max(data['tim])=29.9997, should use len(data['dff'])/exp_p['frameRate']

    Return:
        the modeled calcium
    '''
    
    x = np.arange(0, duration, 1/frameRate)
    y = np.array([Ca_rest]*len(x))
    unfilt = np.array([Ca_rest]*len(x))
    
    
    # 
    for i in range(len(spkT)):
    
        if i < len(spkT)-1:
            ind = np.where(x>=spkT[i])[0][0] # find the first index where x>=one_spike)time
            lastind = np.where(x>=spkT[i+1])[0][0]

            if (lastind-ind) <= 2:
                lastind = ind + 2 # to make 2 spikes at least 3 time points to process

        else: # last spike

            ind = np.where(x>=spkT[i])[0][0] # find the first index where x>=one_spike)time
            lastind = np.where(x>=spkT[i])[0][-1]

            if (lastind-ind) <= 2:
                ind = lastind - 2


        # the time span for this spike
        tspan = x[ind:lastind]
        currentCa = unfilt[ind]

        Y0 = currentCa  # current ca conc following increment due to next spike
        _,ylow = CalciumDecay(Ca_gamma, Ca_rest, Y0, kappaS, Kd, Conc, tspan) # solving ODE for single comp model


        kappa = Kd*Conc/(currentCa+Kd)**2
        Y0 = currentCa + Ca_amp/(1+kappaS+kappa)   # current ca conc following increment due to next spike
        _,yout = CalciumDecay(Ca_gamma, Ca_rest, Y0, kappaS, Kd, Conc, tspan)   # solving ODE for single comp model

        unfilt[ind:lastind] = yout

        # now onset filtering with rising exponential
        # caonset = (1 - np.exp(-(tspan-tspan(1))./Ca_onsettau))
        caonset = (1 - np.exp(-(tspan-spkT[i])/Ca_onsettau))
        caonset[caonset < 0] = 0
        difftmp = yout - ylow
        yout = difftmp*caonset + ylow

        y[ind:lastind] = yout
    
    
    if ifPlot:
        plt.figure(figsize=(20, 2))
        plt.plot(x, y)
        plt.title('Calcium')
        plt.show()
    
    
    return (y)


def Relax2CaRest(t, x, Ca_gamma, Ca_rest, Ca_kappas, Kd, Conc):
    
    '''
    Aim:
        differential equation describing the decay of calcium conc level to resting level 
        in the presence of indicator dye with variable buffering capacity.
        
        self-defined system of ordinary differential equations (ODEs) that one want to solve using solve_ivp()
    '''
    
    dx_dt = -Ca_gamma * (x-Ca_rest)/(1+Ca_kappas+Kd*Conc/(x + Kd)**2)
    
    return(dx_dt)


def CalciumDecay(p_gamma,p_carest,p_cacurrent,p_kappas,p_kd,p_conc,tspan):
    
    '''
    Aim:
        Use ODE45 to solve single-compartment model differential equation
        this function is called in spkTimes2FreeCalcium() in such way:
            _,ylow = CalciumDecay(Ca_gamma,Ca_rest,Y0, kappaS, Kd, Conc, tspan)
        this function is called in SingleFluorTransient() in sucha way:
           [~,lowtmp] = CalciumDecay(ca_p.ca_gamma,ca_p.ca_rest,Y0, ca_p.ca_kappas, exp_p.kd, exp_p.conc, tspan);
        
    Explanation:
        - ODE45: a method in matlab (https://www.mathworks.com/help/matlab/ref/ode45.html). 
                 similari fun in python: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
        - single-compartment model: the simplest way to describe the process of drug distribution 
        and elimination in the body
        
        - solve_ivp: numerically integrates a system of ordinary differential equations given an initial value
    
    '''
    
    t_points = tspan
    t0 = t_points[0]
    y0 = p_cacurrent
    sol = solve_ivp(Relax2CaRest, [t_points[0], t_points[-1]], [y0], 
                    args=(p_gamma, p_carest, p_kappas, p_kd, p_conc), t_eval=t_points)
    
    return(sol.t, sol.y[0])


def sortAbasedonB(arrayP, arrayQ):
    '''
    This part is corresponding to matlab code: q(p) = q in function findClosest(). 
    Here is an example to help explanaing:
        a1 = [0, 1, 2, 3, 4]
        a2 = [2, 4, 3, 1, 0]
        a1[a2] = a1 meaning:
        a1[a2[0]] = a1[0], that is a1[2]=a1[0]=0
        a1[a2[1]] = a1[1], that is a1[4]=a1[1]=1
        etc...
    '''
    
    seqLen = len(arrayP)
    
    qNew = np.zeros(seqLen)
    for i in range(seqLen):
        qNew[arrayP[i]] = arrayQ[i]
    
    
    return(qNew.astype(int))


def findClosest(arrayA, arrayB):
    
    '''
    Aim:
        For each element in arrayA, find the closest in arrayB based on absolute difference
        
    Return:
        Return a list of indices of arrayB to match arrayA
        The distance between select-arrayB and arrayA
    
    Note:
        arrayA, arrayB may of different length. 
    
    '''
    
    
    m = len(arrayA)
    n = len(arrayB)
    
    
    # merge and sort
    arrayAB = np.concatenate((arrayA, arrayB))
    p = np.argsort(arrayAB)
    # p: the indices that would sort an array
    

    q = np.array(list(range(0, m+n))) 
    q = sortAbasedonB(p, q) 
    # the index to reverse the sorting
    # that means: arrayABSort[q] = arrayAB
    # first m from q are the indices to get arrayA
    
    
    t = np.cumsum(p>(m-1)) 
    # p>(m-1) means if each element is from arrayB, T-arrayB
    # cumulated count of elements belong to arrayB
    
    
    r = np.argsort((arrayB))
    # sort arrayB and return index
    
    
    s = t[q[0:m]] 
    # first m from q are the indices to get arrayA
    # for arrayA[0], s[0] ele from arrayB is smaller than it
    # for arrayA[1], s[1] ele from arrayB are smaller than it
    # for arrayA[5], s[5] ele from arrayB are smaller than it
    
    idx = r[np.minimum(s, m-1)] # cap with largest index of arrayA
    # picked ele from arrayB which have smallest dis with arrayA
    
    iux = r[np.maximum(s-1,0)]
    # same with idx, but shift 1 position
    
    
    # and then, by using these two indices as reference, pick the index with smaller distance as final choise
    diff_idx = abs(arrayA-arrayB[idx])
    diff_iux = abs(arrayA-arrayB[iux])

    d = []
    itx = []
    for i in range(m):

        if diff_idx[i] > diff_iux[[i]]:
            d.append(diff_iux[i])
            itx.append(iux[i])
        else:
            d.append(diff_idx[i])
            itx.append(idx[i])

    return(d, itx)


def Fluor2Calcium(f, ca_rest, kd, dffmax):
    
    '''
    Aim:
        Conversion function. Transforms DFF value to calcium concentration absolute value 
    
    '''
    
    caout = (ca_rest + kd*f/dffmax)/(1 - f/dffmax)
    
    
    return(caout)


def SingleFluorTransient(ca_p, exp_p, data, mode, starttim):
    
    '''
    Arguments:
        - ca_p: parameters of elementary (1 AP) calcium transient (calcium dynamics)    
        - data: data and analysis traces
        - mode: 'linDFF' or 'satDFF'
        - starttim: start of the fluorescence transient
    
    Note:
        This function is called in InitPeeling() function at last step (last line of code) in such way:
            [ca_p, exp_p, data] = SingleFluorTransient(ca_p, exp_p, data, ca_p.ca_genmode, 1./exp_p.frameRate)
        This function is also called in Peeling():
            [ca_p, exp_p, data] = SingleFluorTransient(ca_p, exp_p, data, peel_p.spk_recmode, peel_p.nextevt)
        
        Values got changed in the end:
            - ca_p['onsetposition']: from 0 to 1/exp_p['frameRate']
            - data['singleTransient']: from a bunch of 0s to a rise-decay shape
            - ca_p['ca_current']: from 50 to another value, e.g., 41.637
            - exp_p['kappab']: calculated using ca_current, not ca_rest
            - ca_p['ca_amp1']: from 0 to another value, e.g., 30.6490
             
    Return:
        ca_p, exp_p, data
        
    '''

    
    ca_p['onsetposition'] = starttim
    if mode == 'linDFF':
        data['singleTransient'] = np.array([ca_p['offset']]*exp_p['numpnts']).astype('float')
    elif mode == 'satDFF':
        data['singleTransient'] = np.zeros((1,exp_p['numpnts'])).astype('float')
        
        
    ind = data['tim'] >= ca_p['onsetposition']
    firstind = np.where(ind==True)[0][0]
    lastind = np.where(ind==True)[0][-1]

    # start to update single transient shape
    if mode == 'linDFF':
        data['singleTransient'][ind] = ca_p['offset']+ca_p['scale']*(1-np.exp(-(data['tim'][ind]-ca_p['onsetposition'])/ca_p['tauOn']))*(ca_p['A1']*np.exp(-(data['tim'][ind]-ca_p['onsetposition'])/ca_p['tau1'])+ca_p['A2']*np.exp(-(data['tim'][ind]-ca_p['onsetposition'])/ca_p['tau2']))

    elif mode == 'satDFF':

        if lastind - firstind <= 2:
            firstind = lastind-2  # have at least 3 points at end of trace for processing

        if firstind == 0:
            ca_p['ca_current'] = Fluor2Calcium(data['dff'][0], ca_p['ca_rest'], exp_p['kd'], exp_p['dffmax'])  # set to rest when transsient at start of trace
        else:
            ca_p['ca_current'] = Fluor2Calcium(data['dff'][firstind-1], ca_p['ca_rest'], exp_p['kd'], exp_p['dffmax'])  #calculate current, preAP Ca level


        tspan = data['tim'][firstind:lastind+1]

        # decay from pre AP level
        Y0 = ca_p['ca_current']
        _, lowtmp = CalciumDecay(ca_p['ca_gamma'],  ca_p['ca_rest'], Y0, 
                                 ca_p['ca_kappas'], exp_p['kd'], exp_p['conc'], tspan)
        lowdff = Calcium2Fluor(lowtmp,ca_p['ca_rest'], exp_p['kd'], exp_p['dffmax'])

        # decay from post AP level
        exp_p['kappab'] = exp_p['kd']*exp_p['conc']/(ca_p['ca_current']+exp_p['kd'])**2  # recalculate kappab and ca_amp1 
        ca_p['ca_amp1'] = ca_p['ca_amp']/(1+ca_p['ca_kappas']+exp_p['kappab'])
        Y0 = ca_p['ca_current'] + ca_p['ca_amp1']
        _,hightmp = CalciumDecay(ca_p['ca_gamma'], ca_p['ca_rest'],Y0, 
                                 ca_p['ca_kappas'], exp_p['kd'], exp_p['conc'], tspan)
        highdff = Calcium2Fluor(hightmp,ca_p['ca_rest'], exp_p['kd'], exp_p['dffmax'])

        difftmp = highdff - lowdff
        caonset = (1 - np.exp(-(tspan-tspan[1])/ca_p['ca_onsettau']))  # filter with exponential rise 
        data['singleTransient'][firstind:lastind+1] = difftmp*caonset

    else:
        raise Exception('Undefined mode for SingleTransient generation')

        
    return(ca_p, exp_p, data)


def InitPeeling(dff, rate, ifPlot=False):
    
    '''
    Aim & Note:
        Initialization routine.
        This function is  called in such way in Peeling():
            [ca_p, exp_p, peel_p, data] = InitPeeling(peel_dff, peel_rate, args)
    
    Arguments:
        dff: dff of a single neuron
        rate: frame rate
    
    Return: 
        ca_p, exp_p, peel_p, data
    
    '''
    
    
    ca_p = {}
    #--------------------------------------------------------
    # ca_p: parameters of elementary (1 AP) calcium transient
    ca_p['onsetposition'] =0.0    # onset position(s)

    #ca_p.usefreeca = 0           # flag, 1 - use free calcium conc calculations and conversion to DF/F 
    ca_p['ca_genmode'] = 'linDFF' # flag for spike generation mode: 'linDFF' - simple linear DFF, or 'satDFF' - saturating indicator 
    ca_p['ca_onsettau']=0.02      # Ca transient - onset tau (s)
    ca_p['ca_amp']=7600           # Ca transient - total amplitude 1 (nM)
    ca_p['ca_gamma']=400          # Ca transient - extrusion rate (1/s)
    ca_p['ca_amp1']=0             # Ca transient - free Ca amplitude 1  (nM)
    ca_p['ca_tau1']=0             # Ca transient - free Ca tau (s)
    ca_p['ca_kappas']=100         # Ca transient - endogenous Ca-binding ratio 
    ca_p['ca_rest'] = 50          # presumed resting calcium concentration (nM)
    ca_p['ca_current'] = 50       # current calcium concentration (nM)   

    # now parameters for Indicator DF/F(or DR/R) used if 'useFreeCalcium' = 0 otherwise conversion equation is used                            
    ca_p['tauOn']=0.02         # onset tau (s)
    ca_p['offset']=0              # baseline offset (#)
    ca_p['A1']=2.5              # amplitude 1  (#)
    ca_p['tau1']=0.6              # tau1 (s)
    ca_p['A2']=0                # amplitude 2 (#)
    ca_p['tau2']=1.0              # tau2 (s)
    ca_p['integral']=0.0          # integral below curve (#s)
    ca_p['scale']=1.0             # scale factor to scale entire trace (s)
    
    
    exp_p = {}
    #--------------------------------------------------------
    # exp_p: experiment parameters, including dye properties and data acquisition 
    exp_p['numpnts'] = len(dff)    # num points
    exp_p['frameRate'] = rate        # acquisition rate (Hz)
    exp_p['noiseSD'] = 1.2         # noise stdev of DF/F trace (in percent), should be specified by the user
    exp_p['indicator'] = 'OGB-1'   # calcium indicator
    exp_p['dffmax'] = 93           # saturating dff max (in percent)
    exp_p['kd'] = 250              # dye dissociation constant (nM)
    exp_p['conc'] = 50000          # dye total concentration (nM)
    exp_p['kappab'] = exp_p['kd']*exp_p['conc']/(ca_p['ca_rest']+exp_p['kd'])**2   # exogenous (dye) Ca-binding ratio

    if ca_p['ca_genmode'] == 'linDFF':
        pass
    elif ca_p['ca_genmode'] == 'saturatDFF':
        ca_p['ca_amp1'] = ca_p['ca_amp']/(1+ca_p['ca_kappas']+exp_p['kappab'])     # init for consistency
        ca_p['ca_tau1'] = (1+ca_p['ca_kappas']+exp_p['kappab'])/ca_p['ca_gamma']


    peel_p = {}
    #--------------------------------------------------------
    # peel_p: parameters for peeling algorithm
    peel_p['spk_recmode'] = 'linearDFF' # flag,for spike reconstruction mode: 'linearDFF', or 'saturatDFF'  
    peel_p['padding'] = 20              # number of points for padding before and after

    # peel_p['noiseSD'] = 1.4           # expected SD baseline noise level

    peel_p['smtthigh'] = 2.4            # Schmitt trigger - high threshold (multiple of exp_p.noiseSD), 
    peel_p['smttlow'] = -1.2            # Schmitt trigger - low threshold (multiple of exp_p.noiseSD), 
    peel_p['smttbox']= 3                # Schmitt trigger - smoothing box size (in points)
    peel_p['smttmindur']= 0.3           # Schmitt trigger - minimum duration (s)

    '''
     HL: 2012-05-04
     new parameter: max. frames fro smttmindur
     if the frame rate is high, number of frames for smttmindur can be
     large, thereby increasing false negatives
     if smttminFrames is set, use binning to reduce the number of
     frames to this value for high frame rates
     peel_p.smttminFrames = 20
    '''
    peel_p['smttnumevts']= 0      # Schmitt trigger - number of found events
    peel_p['slidwinsiz']= 10.0    # sliding window size - event detection (s)
    peel_p['maxbaseslope']= 0.5   # maximum baseslope #/s
    peel_p['evtfound']=False      # flag - 1: crossing found 
    peel_p['nextevt']=0           # next crossing found (s)
    peel_p['nextevtframe']=0      # next crossing found (frame number)
    peel_p['intcheckwin']=0.5     # window to the right - for integral comparison (s)
    peel_p['intacclevel']=0.5     # event integral acceptance level (0.5 means 50#)
    peel_p['fitonset']=False      # flag - 1: do onset fit, only useful if 1/frameRate <= rise of CacliumTransient
    peel_p['fitwinleft']=0.5      # left window for onset fit (s)
    peel_p['fitwinright']=0.5     # right window for onset fit (s)
    peel_p['negintwin']=0.1       # window to the right - for negativeintegral check(s)
    peel_p['negintacc']=0.5       # negative acceptance level (0.5 means 50#)
    peel_p['stepback']=5.0        # stepsize backwards for next iteration (s)
    peel_p['fitupdatetime']=0.5   # how often the linear fit is updated (s)
    
    
    data = {}
    #--------------------------------------------------------
    # data: data struct 
    
    dff_length = len(dff)
    data['dff'] = dff
    data['peel'] = dff

    if ifPlot:
        # plot currentTransient to get a sense if an event exists
        plt.figure(figsize=(20,2))
        plt.plot(data['peel']) # all zeros
        plt.title('dff to apply peeling on')
        plt.show()
    
    data['intdff'] = np.arange(dff_length)                 # integral curve
    data['temp']   = np.arange(dff_length)                 # temporary wave

    data['freeca']          = np.zeros(exp_p['numpnts'])   # free calcium transient, from which dff will need to be calculated 
    data['singleTransient'] = np.zeros(exp_p['numpnts'])   # fluorescence transient for current AP, will take ca2fluor mode into account
    data['freecamodel']     = np.zeros(exp_p['numpnts'])
    data['model']           = np.zeros(exp_p['numpnts'])
    data['spiketrain']      = np.zeros(exp_p['numpnts'])   # np.zeros((1,exp_p['numpnts'])) for each cell, probably not right
    data['slide']           = np.zeros(exp_p['numpnts'])   # sliding curve, zero corrected

    data['tim'] = np.arange(dff_length)/exp_p['frameRate']
    data['spikes'] = np.zeros(1000)                        # array for found spikes times
    data['numspikes'] = 0                                  # number of spikes found

    # single transient shape
    ca_p, exp_p, data = SingleFluorTransient(ca_p, exp_p, data, ca_p['ca_genmode'], 1/exp_p['frameRate'])
    
    
    return(ca_p, exp_p, peel_p, data)


def overrideFieldValues(dic_toUpdate, dic_toUse):
    
    '''
    Aim:
        update certain key-value pairs in dic_toUpdate using values from dic_toUse
        dic_toUpdate should be a complete verion with all keys, while dic_toUse only contains certain to-be-updateing fields 
        
    Return:
        updated dic_toUpdate
    
    '''
    
    keyList = list(dic_toUse.keys()) 
    for key in keyList:
        dic_toUpdate[key] = dic_toUse[key]
    
    return dic_toUpdate


def PaddingTraces(exp_p, peel_p, data):
    
    '''
    Aim:
        Padding of traces with 0s before and after the working array
        Specifically,  wsiz_of_0z + dff + wsiz_of_0z. That is, 0..000.dff..000..0
        
        This function is called like this in FindNextEvent(): data = PaddingTraces(exp_p, peel_p, data)
    
    Return:
        data
    '''
    
    # positions
    padded_len = exp_p['numpnts']+2*peel_p['padding']
    ori_start = peel_p['padding']
    ori_end = peel_p['padding']+exp_p['numpnts']

    
    # pad on dff
    # create an empty array dff_pad
    data['dff_pad'] = np.zeros(padded_len) # 900+2*300
    # make dff_pad[middle_part] = dff
    data['dff_pad'][ori_start:ori_end] = data['dff'][0:exp_p['numpnts']]


    # repeat padding on peel
    data['peel_pad'] = np.zeros(padded_len)
    data['peel_pad'][ori_start:ori_end] = data['peel'][0:exp_p['numpnts']]


    # update some other fields
    data['slide_pad'] = np.zeros(padded_len)
    data['temp_pad'] = np.zeros(padded_len)

    timPad_s = -peel_p['padding']
    timPad_e = padded_len-peel_p['padding']
    data['tim_pad'] = np.array(list(range(timPad_s, timPad_e)))
    data['tim_pad'] = data['tim_pad']/exp_p['frameRate']

    
    return(data)


def IntegralofCaTransient(ca_p, peel_p, exp_p, data):
    
    '''
    Aim:
        calculate integral for window
        
        This function is called like this in FindNextEvent(): 
            ca_p = IntegralofCaTransient(ca_p, peel_p, exp_p, data)
    
    Arguments:
        - ca_p: parameter for calcium dynamics
        - intvl: window from onset for integral calculation (s)

    Return:
        ca_p
    '''
    
    
    
    if peel_p['spk_recmode'] == 'linDFF':
    
        ca_p['integral'] = ca_p['A1']*(ca_p['tau1']*(1-np.exp(-peel_p['intcheckwin']/ca_p['tau1']))-ca_p['tau1']/(1+ca_p['tau1']/ca_p['tauOn'])*(1-np.exp(-peel_p['intcheckwin']*(1+ca_p['tau1']/ca_p['tauOn'])/ca_p['tau1'])) )
        ca_p['integral'] = ca_p['integral'] + ca_p['A2']*(ca_p['tau2']*(1-np.exp(-peel_p['intcheckwin']/ca_p['tau2'])) - ca_p['tau2']/(1+ca_p['tau2']/ca_p['tauOn'])*(1-np.exp(-peel_p['intcheckwin']*(1+ca_p['tau2']/ca_p['tauOn'])/ca_p['tau2'])) )
        ca_p['integral'] = ca_p['integral'] * ca_p['scale']

        # negative integral for subtraction check
        ca_p['negintegral'] = ca_p['A1']*(ca_p['tau1']*(1-np.exp(-peel_p['negintwin']/ca_p['tau1'])) - ca_p['tau1']/(1+ca_p['tau1']/ca_p['tauOn'])*(1-np.exp(-peel_p['negintwin']*(1+ca_p['tau1']/ca_p['tauOn'])/ca_p['tau1'])) )
        ca_p['negintegral'] = ca_p['negintegral'] + ca_p['A2']*(ca_p['tau2']*(1-np.exp(-peel_p['negintwin']/ca_p['tau2'])) - ca_p['tau2']/(1+ca_p['tau2']/ca_p['tauOn'])*(1-np.exp(-peel_p['negintwin']*(1+ca_p['tau2']/ca_p['tauOn'])/ca_p['tau2'])) )
        ca_p['negintegral'] = ca_p['negintegral'] * -1.0 * ca_p['scale']

        
    elif peel_p['spk_recmode'] == 'satDFF':
        startIdx = min(round(ca_p['onsetposition']*exp_p['frameRate']), 
                   len(data['singleTransient'])-1 )
        stopIdx = min(round((ca_p['onsetposition']+peel_p['intcheckwin'])*exp_p['frameRate']), 
                      len(data['singleTransient']))

        currentTim = data['tim'][startIdx:stopIdx] # current check window
        currentTransient = data['singleTransient'][startIdx:stopIdx]

        ca_p['integral'] = trapezoid(currentTransient, x=currentTim) #scipy.integrate.trapezoid(y, x=None)
        ca_p['integral'] = ca_p['integral'] * ca_p['scale']


        # negative integral for subtraction check
        stopIdx = min(round((ca_p['onsetposition']+peel_p['negintwin'])*exp_p['frameRate']), 
                      len(data['singleTransient']) )
        
        currentTim = data['tim'][startIdx:stopIdx]
        currentTransient = data['singleTransient'][startIdx:stopIdx]

        ca_p['negintegral'] = trapezoid(currentTransient, x=currentTim)
        ca_p['negintegral'] = ca_p['negintegral'] * -1.0 * ca_p['scale']


    else:
        raise Exception ('Error in CaIntegral calculation. Illdefined mode.')

        
    return(ca_p)


def FindNextEvent(ca_p, exp_p, peel_p, data, starttim, ifPlot=False,verbose=False):
    '''
    Aim:
        use the Schmitt trigger to detect the next event
            - firing rate jumps over peel_p['smtthigh']
            - last a certain num of frames 
              that is won't fall below peel_p['smttlow'] in peel_p['smttmindurFrames'] frames
            - Integral > default value in checking period (onset+checkwsiz long)
        
        once found an event:
            - set peel_p['evtfound'] to True
            - record onset frame in peel_p['nextevtframe']
            - record onset time in peel_p['nextevt']
            - and at last, update data['peel'] (not sure why)
            
    Argumnet:
        - exp_p: parameter of data set (either experimental or simulated)
        - alg_p - algorithm parameters/settings
        - starttim - time point for starting search (s)
    
    Return:
        ca_p, peel_p, data
    '''
    
    
    # at the beginnning, set it to False
    peel_p['evtfound'] = False
    
    # precheck
    if (starttim < 0) or (starttim > exp_p['numpnts']/exp_p['frameRate']):
        return (ca_p, peel_p, data)
    
    
    # pad __dff__, __peel__
    wsiz = round(peel_p['slidwinsiz']*exp_p['frameRate'])
    peel_p['padding'] = wsiz
    data = PaddingTraces(exp_p, peel_p, data)


    # get check window size 
    # num of frames from onset to calculate Integral
    checkwsiz = round(peel_p['intcheckwin']*exp_p['frameRate'])
    ca_p['onsetposition'] = 1/exp_p['frameRate']
    ca_p = IntegralofCaTransient(ca_p, peel_p, exp_p, data)


    # start frame
    nstart = round(starttim*exp_p['frameRate']+0.5)+wsiz
    # do linear fit every updateFit(s)/updateFitFrames
    updateFit = peel_p['fitupdatetime']
    updateFitFrames = math.ceil(updateFit*exp_p['frameRate'])
    frameCounter = updateFitFrames+1
    
    
    # start looping each frame to find satisfied onset
    for n in range(nstart, len(data['peel_pad'])-wsiz): 
    
        if frameCounter > updateFitFrames:

            frameCounter = 0 
            currentwin = data['peel_pad'][n-wsiz:n] #[n-wsiz:n-1] in matlab = wsiz(300) values
            currenttim = data['tim_pad'][n-wsiz:n]

            linefit = np.polyfit(x=currenttim, y=currentwin, deg=1)
            tmpslope = linefit[0]

            if tmpslope > peel_p['maxbaseslope']: # peel_p['maxbaseslope']=0.5
                tmpslope = peel_p['maxbaseslope']
            elif tmpslope < -peel_p['maxbaseslope']:
                tmpslope = -peel_p['maxbaseslope']  
        else:
            frameCounter = frameCounter + 1


        currentoffset = tmpslope*data['tim_pad'][n-1] + linefit[1]


        # Schmitt trigger Loop
        if data['peel_pad'][n]-currentoffset > peel_p['smtthigh'] : #just over the lower bound

            # just to see if the signal has not reached the end of dff
            if n + peel_p['smttmindurFrames'] <= len(data['peel_pad']): 
                currentDff = data['peel_pad'][n:n+peel_p['smttmindurFrames']] # from signal onset -- 9 frames later
            else:
                currentDff = data['peel_pad'][n:]


            # if any(currentDff <= peel_p.smttlow) 
            # that is if currentDFF decay lower below the point that we thing signal ends
            # that is to say, this period can no be seens as an event, too short
            if len( np.argwhere(currentDff <= peel_p['smttlow']) ) > peel_p['smttlowMinEvents'] :

                n = n + np.argwhere(currentDff <= peel_p['smttlow'])[-1]
                if n > len(data['peel_pad'])-wsiz : # hit the end of dff trace
                    break

                # frameCounter --> signal last
                frameCounter = frameCounter + np.argwhere(currentDff <= peel_p['smttlow'])[-1]
                continue # return to the beginning of the for loop



            # check if an event exists
            data['slide_pad'] = data['peel_pad'] - currentoffset
            data['temp_pad'] = tmpslope*data['tim_pad'] + linefit[1] - currentoffset
            data['slide_pad'] = data['slide_pad'] - data['temp_pad']

            currentIntegral = trapezoid(data['slide_pad'][n:n+checkwsiz], x=data['tim_pad'][n:n+checkwsiz])

            if peel_p['spk_recmode'] == 'linDFF':
                pass
            elif peel_p['spk_recmode'] == 'satDFF':
                ca_p['onsetposition'] = (n-wsiz)/exp_p['frameRate']
                ca_p = IntegralofCaTransient(ca_p, peel_p, exp_p, data)


            if currentIntegral > (ca_p['integral']*peel_p['intacclevel']):

                peel_p['evtfound'] = True
                if verbose:
                    print('  FindNextEvent() Found event at n='+str(n-wsiz))
                
                # plot the padded check period to get a sense if an event exists
                if ifPlot:
                    plt.figure(figsize=(20,2))
                    plt.plot(data['slide_pad'][n-10:n+checkwsiz+10], label='slide_pad')
                    plt.plot(data['dff'][n-10-wsiz:n+checkwsiz+10-wsiz].to_numpy(), label='dff')
                    plt.title('dff + slide_pad, +10 frame around')
                    plt.legend()
                    plt.show()

                break
    
    
    # record onset frame & time
    if peel_p['evtfound']:
        peel_p['nextevtframe'] = n-wsiz
        peel_p['nextevt'] = (n-wsiz) / exp_p['frameRate']

        
    data['peel'] = data['peel_pad'][ peel_p['padding'] : peel_p['padding']+exp_p['numpnts'] ]

    
    return(ca_p, peel_p, data)


def modelCalciumTransient(t,onsettime,onsettau,amp1,tau1):
    
    '''
    The shape of 1 calcium transient
    '''
    
    
    offset=0
    y = np.array([offset]*len(t)).astype(float)

    indx = np.argwhere(t>onsettime)
    y[indx] = offset + (1-np.exp(-(t[indx]-onsettime)/onsettau)) * (amp1*np.exp(-(t[indx]-onsettime)/tau1))
    
    return(y)


def PeelingOptimizeSpikeTimes(dff, spkTin, lowerT, upperT, rate, tauOn, A1, tau1, optimMethod='L-BFGS-B', ifPlot=False):
    '''
    Aim:
        optimization of spike times found by Peeling algorithm
        minimize the sum of the residual squared while several optimization algorithms 
        are implemented (see below), we have only used pattern search. 
        Other algorithms are only provided for convenience and are not tested sufficiently.

    Analogy:
        Imagine you have a stack of transparent bell curves (transients) placed at initial spike times. 
        You shift them left or right within a small window, trying to make the sum of curves align with the measured trace (dff).
        That’s what the optimizer does!
    
    Python:
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
    
    Note:
        - optimMethod should choose from: 'Nelder-Mead', 'L-BFGS-B', 'Nelder-Mead', 'TNC', 'SLSQP', 'Powell', 'trust-constr', 'COBYLA'
        - default is 'Nelder-Mead', which is pattern search method


    Return:
        spkTout
        output
        
    '''
    
    # parameters
    t = np.array(list(range(len(dff))))/rate
    modelTransient = modelCalciumTransient(t, t[1], tauOn, A1, tau1)
    
    # reconstruct dff using modelTransient  and spikeTimes
    model = reconstructModel(t, spkTin, modelTransient)
    # calculate residual
    residual = dff - model
    resInit = sum(residual**2)


    # Start Optimization
    # that is use searching local minimum method to find the spkTimes that make the residual minimum
    np.random.seed(555)   # Seeded to allow replication.

    x0 = spkTin    # Initial guess.
    params = (dff, rate, tauOn, A1, tau1) # parameters for objectiveFunc()

    lbound = spkTin - lowerT
    ubound = spkTin + upperT
    bnds = [(lbound[i], ubound[i]) for i in range(len(spkTin))]
    res = optimize.minimize(objectiveFunc, x0, args=params, method=optimMethod, bounds=bnds)

    '''
    Example res:
        message: CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL
    success: True
    status: 0
        fun: 2208.6704057308116
            x: [ 1.267e+00  1.213e+01  1.453e+01]
        nit: 0
        jac: [ 0.000e+00  0.000e+00  0.000e+00]
        nfev: 4
        njev: 1
    hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
    '''

    if res.success and res.fun<resInit: # if success and got smaller residual
        spkTout = res.x
    else:
        spkTout = spkTin
        print('Optimization did not improve residual. Keeping input spike times.')


    # # plot to see 
    # modelOut = reconstructModel(t, spkTout, modelTransient)
    # if ifPlot:
    #     plt.figure(figsize=(20,2))
    #     plt.plot(model,     color='r', lw=1, label='pre model')
    #     plt.plot(modelOut,  color='green', lw=1, label='opt model')
    #     plt.plot(dff,       color='k', lw=4, label='dff',       alpha=0.2)
    #     plt.plot(dff-model, color='b', lw=1, label='dff-model', alpha=0.4)
    #     plt.title('SpikeTimes After Optimization')
    #     plt.show()


    return(spkTout, residual)

def objectiveFunc(spkTin, *params):
    
    '''
    This funciton is the objective function used in optimize.minimize() in PeelingOptimizeSpikeTimes(), 
    which is an algorithm to find minimum of function using simulated annealing algorithm
    x = simulannealbnd(fun,x0,lb,ub,options) minimizes with the optimization options specified in options. 

    '''
    
    dff, rate, tauOn, A1, tau1 = params
    t = np.array(list(range(len(dff))))/rate
    #modelTransient = spkTimes2Calcium(0, tauOn, A1, tau1, 0, 0, rate, len(dff)/rate)
    modelTransient = modelCalciumTransient(t, t[1], tauOn, A1, tau1)

    model = reconstructModel(t, spkTin, modelTransient)

    residual = dff - model
    residual = sum(residual**2)
    
    return(residual)

def reconstructModel(t, spkTimes, modelTransient):
    
    '''
    Aim:
        locate the spikTimes along t, convolve spikes and modelTransient
        This function is called in PeelingOptimizeSpikeTimes()
        modelTransient: a typical transient shape
    
    '''
    
    # spkVector = np.zeros(len(t))
    # for i in range(len(spkTimes)):

    #     spkTt_diff = list(abs(spkTimes[i]-t))
    #     idx = spkTt_diff.index(min(spkTt_diff))
    #     spkVector[idx]= spkVector[idx]+1
    # model = np.convolve(spkVector, modelTransient) 
    # model = model[0:len(t)]


    # Vectorized version (faster)
    spkVector = np.zeros(len(t))
    idxs = np.searchsorted(t, spkTimes)
    idxs = idxs[idxs < len(t)]
    np.add.at(spkVector, idxs, 1)
    model = np.convolve(spkVector, modelTransient, mode='full')[:len(t)]

    return(model)



def PeelingOptimizeSpikeTimesSaturation(dff,spkTin,lowerT,upperT,ca_amp,ca_gamma,ca_onsettau,ca_rest,ca_kappas,
                                        kd,conc,dffmax,frameRate,dur,optimMethod='Nelder-Mead',ifPlot=False):
    '''
    Aim:
        optimization of spike times found by Peeling algorithm
        minimize the sum of the residual squared while several optimization algorithms 
        are implemented (see below), we have only used pattern search. 
        Other algorithms are only provided for convenience and are not tested sufficiently.
    
    Python:
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
    
    Note:
        - optimMethod should choose from: 'Nelder-Mead', 'L-BFGS-B', 'Nelder-Mead', 'TNC', 'SLSQP', 'Powell', 'trust-constr', 'COBYLA'
        - default is 'Nelder-Mead', which is pattern search method

    Return:
        spkTout
        output
        
    '''
    
    # model including saturation effect
    ca = spkTimes2FreeCalcium(spkTin, ca_amp, ca_gamma, ca_onsettau, ca_rest, ca_kappas, kd, conc, frameRate, dur)
    modeltmp = Calcium2Fluor(ca, ca_rest, kd, dffmax)
    model = modeltmp[0:len(dff)]

    # calculate residual
    residual = dff - model
    resInit = sum(residual**2)
    
    
    # Start Optimization
    np.random.seed(555)   # Seeded to allow replication.

    x0 = spkTin    # Initial guess.
    params = (dff, ca_rest, ca_amp, ca_gamma, ca_onsettau, ca_kappas, kd, conc, dffmax, frameRate, dur)

    lbound = spkTin - lowerT
    ubound = spkTin + upperT
    bnds = [(lbound[i], ubound[i]) for i in range(len(spkTin))]

    res = optimize.minimize(objectiveFuncSaturation, x0, args=params, method=optimMethod, bounds=bnds)
    if res.success and res.fun<resInit: # if success and got smaller residual
        spkTout = res.x
    else:
        spkTout = spkTin
        # print('Optimization did not improve residual. Keeping input spike times.')
    
    # reconstruct model using spkTout
    caOut = spkTimes2FreeCalcium(spkTout, ca_amp, ca_gamma, ca_onsettau, ca_rest, ca_kappas, kd, conc, frameRate, dur)
    modeltmp = Calcium2Fluor(caOut, ca_rest, kd, dffmax)
    modelOut = modeltmp[0:len(dff)]

    if ifPlot:
        plt.figure(figsize=(20,2))
        plt.plot(model,     color='r', lw=1, label='pre model')
        plt.plot(modelOut,  color='green', lw=1, label='opt model')
        plt.plot(dff,       color='k', lw=4, label='dff',       alpha=0.2)
        plt.plot(dff-model, color='b', lw=1, label='dff-model', alpha=0.4)
        plt.title('SpikeTimes After Optimization')
        plt.legend(loc=1)

        plt.show()

    return(spkTout, res)

def objectiveFuncSaturation(spkTin, *params):
    
    '''
    This funciton is the objective function used in optimize.minimize() in PeelingOptimizeSpikeTimesSaturation(), 
    which is an algorithm to find minimum of function using simulated annealing algorithm
    x = simulannealbnd(fun,x0,lb,ub,options) minimizes with the optimization options specified in options. 

    '''
    
    dff, ca_rest, ca_amp, ca_gamma, ca_onsettau, ca_kappas, kd, conc, dffmax, frameRate, dur = params
    ca = spkTimes2FreeCalcium(spkTin,ca_amp,ca_gamma,ca_onsettau,ca_rest,ca_kappas,kd,conc,frameRate,dur)
    modeltmp = Calcium2Fluor(ca,ca_rest,kd,dffmax)
    model = modeltmp[0:len(dff)]

    residual = dff - model
    residual = sum(residual**2)
    
    return(residual)


def Peeling(dff, rate, ca_p, exp_p, peel_p, optimMethod='Nelder-Mead', lowerT=0.5, upperT=0.5):
    
    '''
    Aim:
        This is the main routine of the peeling algorithm
        This function is called in such a way in doPeel():
            ca_p, peel_p, data = Peeling(dff[m,:], frameRate, ca_p, exp_p, peel_p)
        
    Arguments:
        - dff: a neuron's dff
        - rate: frame rate
        - ca_p: parameters of elementary (1 AP) calcium transient (calcium dynamics)  
        - exp_p: experiment/simulated parameters, including dye properties and data acquisition 
        - peel_p: parameters for peeling algorithm
        - optimMethod: method used to get x local minimum of objective function. see PeelingOptimizeSpikeTimes() for more details.
        - lowerT/upperT: search space of x in optimize.minimize()
    
    Return:
    
    '''
    maxRate_peel = float('inf')
    if rate > maxRate_peel:
        peel_rate = maxRate_peel
        fit_rate = rate
        peel_dff = np.array([])
    else:
        peel_rate = rate
        fit_rate = rate
        peel_dff = dff
    
    
    # call InitPeeling() to initialize parameter dictionaries
    ca_pInit, exp_pInit, peel_pInit, data = InitPeeling(peel_dff, rate)

    # update _pInit with values set in doPeel() part
    ca_p = overrideFieldValues(ca_pInit, ca_p)
    exp_p = overrideFieldValues(exp_pInit, exp_p)
    peel_p = overrideFieldValues(peel_pInit, peel_p)

    peel_p['smttmindurFrames'] = math.ceil(peel_p['smttmindur']*exp_p['frameRate'])
    peel_p['smttlowMinEvents'] = 1
    nexttim = 0 #1/exp_p['frameRate']
    #dur = len(data['dff'])/exp_p['frameRate']


    # find first event
    [ca_p, peel_p, data] = FindNextEvent(ca_p, exp_p, peel_p, data, nexttim)
    if peel_p['evtfound']:
        data['numspikes'] = data['numspikes'] + 1
        data['spikes'][data['numspikes']] = peel_p['nextevt'] # the i-th spike happening time
        ca_p, exp_p, data = SingleFluorTransient(ca_p, exp_p, data, peel_p['spk_recmode'], peel_p['nextevt'])
        data['model'] = data['model'] + data['singleTransient']

    # now data['numspikes']=1
    # while loop to exam 1st event and find remaining events
    maxiter = 3000
    iteration = 0
    nexttimMem = float('inf')
    nexttimCounter = 0
    timeStepForward = 2/exp_p['frameRate']

    while peel_p['evtfound']:
        
        # check integral after subtracting Ca transient
        if peel_p['spk_recmode'] == 'linDFF':
            pass
        elif peel_p['spk_recmode'] == 'satDFF':
            ca_p['onsetposition'] = peel_p['nextevt']
            ca_p = IntegralofCaTransient(ca_p, peel_p, exp_p, data)
        
        
        # remove the first event
        dummy = data['peel'] - data['singleTransient'] 
        # exam the previous spike valid or not
        # in 1st run, exam the spike before the while loop
        spk_timDiff = list(abs(data['tim'] - data['spikes'][data['numspikes']]))
        startIdx = spk_timDiff.index(min(spk_timDiff))
        spk_checkTimDiff = list(abs( data['tim'] - (data['spikes'][data['numspikes']]+peel_p['intcheckwin'])))
        stopIdx = spk_checkTimDiff.index(min(spk_checkTimDiff))
        # print('spikeTime at', data['spikes'][data['numspikes']], 'spike starts at', startIdx, ', and check window ends at', stopIdx)
        
        # clean up indices
        if startIdx < stopIdx:
            currentTim = data['tim'][startIdx:stopIdx]
            currentPeel = dummy[startIdx:stopIdx]
            currentIntegral = trapezoid(currentPeel, x=currentTim)# integral after one peel
        else:
            # if enters here, startIdx is the last data point and we should not accept it as a spike
            currentIntegral = ca_p['negintegral']*peel_p['negintacc']
        
        # this step is to see if that spike is real
        # after substract that spike from peel resulting dummy
        # if integral_of_spike_checkWindow in dummy <= negative intergral, then this spike is removed in else part
        # that is this spike can not be considered as a real spike
        if currentIntegral > (ca_p['negintegral']*peel_p['negintacc']):
            # print('dummy_checkWin Integral>negIntegral, real spike, peel off singleTransient')
            
            data['peel'] = data['peel'] - data['singleTransient'] # peel off happens here
            nexttim = data['spikes'][data['numspikes']] - peel_p['stepback']
            if nexttim < 0:
                nexttim = 0 #1/exp_p['frameRate'] 
            # print('step back to nexttim='+str(round(nexttim,4))+' and check again.')
        else:
            # print('dummy_checkWin Integral<=negIntegral. Revoke spike, numspikes, and singleTransient in data.model')
            data['spikes'][data['numspikes']] = 0
            data['numspikes'] = data['numspikes']-1
            data['model'] = data['model'] - data['singleTransient']
            nexttim = peel_p['nextevt'] + timeStepForward
            # print('step forward to nexttim='+str(round(nexttim,4))+' and check event')

        
        # find next event
        peel_p['evtaccepted'] = False
        ca_p, peel_p, data = FindNextEvent(ca_p, exp_p, peel_p, data, nexttim)
        if peel_p['evtfound']:
            data['numspikes'] = data['numspikes'] + 1
            data['spikes'][data['numspikes']] = peel_p['nextevt']
            ca_p, exp_p, data = SingleFluorTransient(ca_p, exp_p, data, peel_p['spk_recmode'], peel_p['nextevt'])
            data['model'] = data['model'] + data['singleTransient']
        else:
            break
        

        iteration = iteration+1
        if nexttim == nexttimMem:
            nexttimCounter = nexttimCounter + 1
        else:
            nexttimMem = nexttim
            nexttimCounter = 0
        
        if nexttimCounter > 50:
            nexttim = nexttim + timeStepForward 
        
        if iteration > maxiter:
            # warning('Reached maxiter (#1.0f). nexttim=#1.2f. Timeout!',maxiter,nexttim), save, error('Covergence failed!')
            break
        
        
        # print('-----------------')
        # print()
    

    
    print("Total iterations:", iteration)
    print("Total spikes:", data['numspikes'])
    print("len(data['spikes']):", len(data['spikes']))
    # only keep the spike-happening-times 
    # E.g., make the data['spikes'] from (0, 1.1, 2.1, 3.4, 0, 0, ....0) to (1.1, 2.1, 3.4)
    if len(data['spikes']) > data['numspikes']:
        data['spikes'] = data['spikes'][1:data['numspikes']+1] 
        
    
    # optimization of reconstructed spike times to improve timing
    optMethod = optimMethod
    if len(data['spikes']) and peel_p['optimizeSpikeTimes']:
        
        if peel_p['spk_recmode'] == 'linDFF':
            spikes,_ = PeelingOptimizeSpikeTimes(data['dff'], data['spikes'], lowerT, upperT,
                                                 exp_p['frameRate'], ca_p['tauOn'], ca_p['A1'], ca_p['tau1'], optMethod)
        
        elif peel_p['spk_recmode'] == 'satDFF':
            spikes,_ = PeelingOptimizeSpikeTimesSaturation(data['dff'], data['spikes'], lowerT, upperT,
                                                           ca_p['ca_amp'], ca_p['ca_gamma'], ca_p['ca_onsettau'],
                                                           ca_p['ca_rest'], ca_p['ca_kappas'], 
                                                           exp_p['kd'], exp_p['conc'], exp_p['dffmax'], exp_p['frameRate'],
                                                           len(data['dff'])/exp_p['frameRate'], optMethod)
        else:
            raise Exception('Undefined mode')
        
        data['spikes'] = spikes

    
    # loop to create spike train vector from spike times
    data['spiketrain'] = np.zeros(len(data['dff']))
    for i in range(len(data['spikes'])):
            
        spk_timDiff = list(abs(data['spikes'][i]-data['tim']))
        idx = spk_timDiff.index(min(spk_timDiff))
        data['spiketrain'][idx] = data['spiketrain'][idx]+1
    

    # re-derive model and residuals after optimization
    if peel_p['spk_recmode'] == 'linDFF':
        modelTransient = spkTimes2Calcium(0,ca_p['tauOn'], ca_p['A1'], ca_p['tau1'], ca_p['A2'], ca_p['tau2'],
                                        exp_p['frameRate'], len(data['dff'])/exp_p['frameRate'] )
        data['model'] = np.convolve(data['spiketrain'], modelTransient)
        data['model'] = data['model'][0:len(data['tim'])]
    elif peel_p['spk_recmode'] == 'satDFF':
        modeltmp = spkTimes2FreeCalcium(data['spikes'], ca_p['ca_amp'], ca_p['ca_gamma'], ca_p['ca_onsettau'],
                                    ca_p['ca_rest'], ca_p['ca_kappas'],
                                    exp_p['kd'], exp_p['conc'], exp_p['frameRate'], len(data['dff'])/exp_p['frameRate'] )
        data['model'] = Calcium2Fluor(modeltmp, ca_p['ca_rest'], exp_p['kd'], exp_p['dffmax'])
        
    data['peel'] = data['dff'] - data['model']


    # plot to see peel results
    if peel_p['doPlot']:
        plt.figure(figsize=(20,2))
        plt.plot(data['dff'], alpha=0.4, lw=2, label='dff (calcium)')
        plt.plot(data['peel'], label='peel (residual)')
        plt.plot(data['spiketrain'], lw=2, label='spikeTrain (unverified putative APs)') #unverified putative action potential

        plt.legend(loc=1)
        plt.show()
    
    
    return(ca_p, peel_p, data)


def doPeel(dff,frameRate,ca_genmode,spk_recmode,A1,tau1,A2,tau2,tauOn,ca_rest,ca_amp,ca_gamma,
           ca_onsettau,ca_amp1,ca_tau1,ca_kappas,dffmax,kd,kappab,noiseSD,schmitt,peel_p,optimMethod='Nelder-Mead', lowerT=0.5, upperT=0.5):
    
    '''
    Aim:
        Get parameters ready to be put into Peeling() function
        Note that dff should be in shape(num_neurons, downSampledSize), e.g., (1, 900)
        This function is called in modelCalcium()
        
    Return:
        spikePredict
        peel
    '''
    
    
    # create two dictionaries for parameters, which will be used as input to Peeling() function
    ca_p = {}
    exp_p = {}
    # peel_p = peelOpts['peel_p'] as input in function
    

    cellNo  =  dff.shape[0]

    # to store results for each neuron
    peel =  [[] for i in range(cellNo)]
    spikePredict =  [[] for i in range(cellNo)]
    
    # loop through each neuron
    for m in range(cellNo):

        # update some parameters in ca_p and exp_p dictionary
        ca_p['ca_genmode'] = ca_genmode

        ca_p['A1'] = A1[m]
        ca_p['tau1'] = tau1[m]
        ca_p['A2'] = A2[m]
        ca_p['tau2'] = tau2[m]
        ca_p['tauOn']  =  tauOn[m]

        ca_p['ca_amp'] =  ca_amp[m]
        ca_p['ca_amp1'] =  ca_amp1[m]
        ca_p['ca_tau1'] =  ca_tau1[m]
        ca_p['ca_rest'] =  ca_rest[m]
        ca_p['ca_gamma'] =  ca_gamma[m]
        ca_p['ca_kappas'] =  ca_kappas[m]
        ca_p['ca_onsettau'] =  ca_onsettau[m]


        exp_p['dffmax'] = dffmax[m]
        exp_p['noiseSD']  =  noiseSD[m]
        exp_p['kd'] = kd
        exp_p['kappab'] = kappab


        # if ca_p['ca_genmode'] == 'linDFF':
        #     PeakA = A1*(tau1/tauOn*(tau1/tauOn+1)**(-(tauOn/tau1+1)))
        # elif ca_p['ca_genmode'] == 'satDFF':
        #     ca_p['ca_amp1'] = ca_p['ca_amp']/(1+ca_p['ca_kappas']+exp_p['kappab']) 
        #     ca_p['ca_tau1'] = (1+ca_p['ca_kappas']+exp_p['kappab'])/ca_p['ca_gamma']
        #     PeakCa = ca_p['ca_amp1'] + ca_p['ca_rest']
        #     PeakDFF = Calcium2Fluor(PeakCa, ca_p['ca_rest'], exp_p['kd'], exp_p['dffmax'])
        #     PeakA = PeakDFF * (ca_p['ca_tau1']/ca_p['tauOn']*(ca_p['ca_tau1']/ca_p['tauOn']+1)**(-(ca_p['tauOn']/ca_p['ca_tau1']+1)))
        # else:
        #     raise Exception('Calcium trace generation mode illdefined. Chose from linDFF, satDFF.')

        # snr = PeakA/exp_p['noiseSD'] # not used
        # this part seems to be redundant, snr is the values snr[m]
        # exp_p['noiseSD'] is calculated by peakA/snr, pealA is calculated in the exactly same way

    
        peel_p['spk_recmode'] = spk_recmode
        peel_p['smtthigh'] = schmitt[0]*exp_p['noiseSD']
        peel_p['smttlow'] = schmitt[1]*exp_p['noiseSD']

        peel_p['smttmindur']= schmitt[2]
        peel_p['slidwinsiz'] = 10.0
        peel_p['fitupdatetime']=0.5

        peel_p['doPlot'] = False

        # call Peeling()
        ca_p, peel_p, data = Peeling(dff[m,:], frameRate, ca_p, exp_p, peel_p, optimMethod=optimMethod, lowerT=lowerT, upperT=upperT)

        spikePredict[m] = data['spikes']
        peel[m] = data['peel']

        
    return(spikePredict, peel, ca_p, peel_p, exp_p, data)













# Other Functions
#========================================================================================
# def filterNeuron_cv2Thresh(A, randomNeurons, lowerThres=30):
    
#     '''
#     For each spatial component, only keep the brightest one
#     remove any remeaning residual fromt the background
    
#     '''
    
#     # A_simThresh = A[:,randomNeurons] 
#     vs = []; cs = []
#     A_simThresh = np.asarray([[0]*262144]*10).T
#     for i in range(len(randomNeurons)):
#         neuronIndex = randomNeurons[i]
#         aNeuron = A[:, neuronIndex]
#         aNeuron_norm = np.uint8(aNeuron * (aNeuron > 0) * 255)
#         # apply threshold
#         ret,sim_thresh2 = cv2.threshold(aNeuron_norm,lowerThres,255,cv2.THRESH_BINARY)
        
#         A_simThresh[:, i] = sim_thresh2.flatten()
#         v, c = np.unique(A_simThresh[:, i], return_counts=True)
#         vs.append(v); cs.append(c)
    

#     return(A_simThresh, vs, cs)

def deconvolve_df(df, ifSave=True, savePath=None, baseName=None, ifBinary=False, threshold=0.5, frameRate=30):
    
    '''
    Aim:
        deconvolute trace for each neuron & save
    
    Note:
        used function: https://github.com/j-friedrich/OASIS/blob/master/oasis/functions.py#L110
        c: nferred denoised fluorescence signal at each timebin
        b: fluorescence baseline
        s: spikes

    Return:
        df_denoised(b+c), df_spikes(s)

    '''
    
    
    df_spikes = df.copy(); df_denoised = df.copy();  spkTimes_dic = {}
    columns = df.columns
    Frames, n_neuron = df.shape

    # start deconvolution
    if Frames != 0 and n_neuron != 0:

        # get dfarray
        df_array = df.to_numpy()

        # get decovoluted trace for each neuron
        for i in range(n_neuron):
            y = df_array[:,i][~np.isnan(df_array[:,i])]
            c, s, b, g, lam = deconvolve(y, penalty=1)
            df_denoised[columns[i]] = b+c
            if ifBinary:
                binary_s = (s > threshold).astype(int)
                df_spikes[columns[i]] = binary_s
                times_s = np.where(binary_s==1)[0]/frameRate
                spkTimes_dic[columns[i]] = times_s #np.floor(times_s * samplingRate).astype(int)
            else:
                df_spikes[columns[i]] = s
                spkTimes_dic[columns[i]] = s
        
        spkTimes_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in spkTimes_dic.items()]))

        # save to path
        if ifSave:
            df_denoised.to_csv(os.path.join(savePath, baseName+'-D.csv'))
            df_spikes.to_csv(os.path.join(savePath, baseName+'-S.csv'))
            spkTimes_df.to_csv(os.path.join(savePath, baseName+'-STimes.csv'))
        
        return (df_denoised, df_spikes, spkTimes_df)
            
    else:
        raise Exception("Empty DataFrame!")
    









# Plot Functions
#========================================================================================
def plot_peelResults(cellNum,x,spkTimes,spikePredict,dff,dffRecon,dffC='grey',dffRecC='blue', ifSave=False,savePath=None,filename=None):
    
    '''
    plot ground truth, peel results and reconstruction on one plot
    
    '''
    
    DFFResMax = dff[:].max().max()
    
    if cellNum == 0:
        m = 0
        
        currentDFF = dff[m, :]
        currentDFFRecon = dffRecon[m, :]
            
        plt.figure(figsize=(20, 3))
        plt.plot(x,currentDFF, c=dffC, alpha=0.4, lw=6, label='DFF Simulation')
        plt.plot(x,currentDFFRecon, c=dffRecC, lw=2, label='DFF Reconstruction')
        
        # plot spkTimes, groundtruth
        for n in range(len(spkTimes[m])):
            if n==0:
                plt.errorbar(x=spkTimes[m][n],label='spk groundTruth',y=max(currentDFF)+DFFResMax/5,yerr=1,c='black',elinewidth=2)
            else:
                plt.errorbar(x=spkTimes[m][n],y=max(currentDFF)+DFFResMax/5,yerr=1,c='black',elinewidth=2)
            
        # plot spkTimes, predicted
        for n in range(len(spikePredict[m])):
            if n==0:
                plt.errorbar(x=spikePredict[m][n],label='spk Predicted',y=max(currentDFF)+DFFResMax,yerr=1,c='blue',elinewidth=2)
            else:
                plt.errorbar(x=spikePredict[m][n],y=max(currentDFF)+DFFResMax,yerr=1,c='blue',elinewidth=2)

        plt.legend(loc=1)
    
    elif cellNum > 1:
        fig, ax = plt.subplots(cellNum,1, figsize=(20,2*cellNum),facecolor='w', edgecolor='k')
        for m in range(cellNum):
            label='neuron'+str(m)
            currentDFF = dff[m, :]
            currentDFFRecon = dffRecon[m, :]
            
            if m==0 or m==cellNum-1:
                ax[m].plot(x,currentDFF, c=dffC, alpha=0.4, lw=6, label='DFF Simulation')
                ax[m].plot(x,currentDFFRecon, c=dffRecC, lw=2, label='DFF Reconstruction')
            else:
                ax[m].plot(x,currentDFF, c=dffC, alpha=0.4, lw=6, label=label)
                ax[m].plot(x,currentDFFRecon, c=dffRecC, lw=2)
            ax[m].legend(loc=1) 
            
            # plot spkTimes, groundtruth
            for n in range(len(spkTimes[m])):
                if n==0 and (m==0 or m==cellNum-1):
                    ax[m].errorbar(x=spkTimes[m][n],label='spk groundTruth',y=max(currentDFF)+DFFResMax/5,yerr=1,c='black',elinewidth=2)
                else:
                    ax[m].errorbar(x=spkTimes[m][n],y=max(currentDFF)+DFFResMax/5,yerr=1,c='black',elinewidth=2)
                ax[m].legend(loc=1) 
                
            # plot spkTimes, predicted
            for n in range(len(spikePredict[m])):
                if n==0 and (m==0 or m==cellNum-1):
                    ax[m].errorbar(x=spikePredict[m][n],label='spk Predicted',y=max(currentDFF)+DFFResMax,yerr=1,c='blue',elinewidth=2)
                else:
                    ax[m].errorbar(x=spikePredict[m][n],y=max(currentDFF)+DFFResMax,yerr=1,c='blue',elinewidth=2)
                ax[m].legend(loc=1) 
        
    else:
        raise Exception('No neuron in this model')
        
        
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_neuronTrace_DFs(dfs,labels,colors,alphas,lws,ifSave=False,savePath=None,filename=None ):
    
    '''
    Aim:
        plot neuron firing rate from 2 different dataframes on one plot
        DFF, Denoised, Deconvoluted

    '''
   
    f, n = dfs[0].shape
    fig, ax = plt.subplots(n,1, figsize=(20,2*n),facecolor='w', edgecolor='k')

    for nIndex in range(n):
    
        subTitle='neuron_'+str(nIndex)

        for i in range(len(labels)):
            ax[nIndex].plot(dfs[i].iloc[:, nIndex], c=colors[i], alpha=alphas[i], lw=lws[i],label=labels[i])
            ax[nIndex].set_title(subTitle)
            ax[nIndex].legend(loc=1)
    
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_selectNeuronShape(A_simThresh,randomNeurons,rows=2,cols=5,figsize=(20, 10),ifTranspose=False,ifSave=False,savePath=None,filename=None ):
    
    '''
    plot select neuron to check neuron shape
    Those selected neuron will be used to dot with simulated neuron activity
    '''
    
    # plot select neurons' shape
    fig, ax = plt.subplots(rows, cols, figsize=figsize,facecolor='w', edgecolor='k')
    neuronNum = 0
    for r in range(rows):
        for c in range(cols):
            ax[r, c].set_title('neuron'+str(randomNeurons[neuronNum]))
            if ifTranspose:
                ax[r, c].imshow((A_simThresh[:, neuronNum]).reshape(512, 512).T)
            else:
                ax[r, c].imshow((A_simThresh[:, neuronNum]).reshape(512, 512))
            neuronNum += 1  

    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()















