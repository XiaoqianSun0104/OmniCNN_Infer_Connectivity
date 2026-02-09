#!/usr/bin/env python
# modelCalcium.py
#
# Author: Xiaoqian Sun, 07/2023 
#
# Simulation object
# Ref: https://github.com/flatironinstitute/CaImAn/blob/main/caiman/source_extraction/cnmf/cnmf.py


# Import Packages
#========================================================================================
import os 
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import modelCa2Params
from peelingFunctions import *



# Class
#========================================================================================
class ModelCalcium(object):
    '''
    Simulation process
        - spike generation
        - spike to calcium to fluorescence (linear/saturation)
        - add Gaussian noise to fluorescence
        - subsampling 
        - correct baseline
        - plot
    '''


    def __init__(self, x=None,  
                 spk_recmode = 'linDFF',dur = 30, offset=1, doPlot=True, doVectorized=False, cellNum=1,
                 spikeRate = np.array([0.8]), ca_genmode = 'linDFF', ca_amp = np.array([7600]), ca_amp1 = np.array([0]),
                 ca_tau1 = np.array([0]), ca_rest = np.array([50]), ca_gamma = np.array([400]), ca_kappas = np.array([100]),
                 ca_onsettau = np.array([0.02]), kd = 250, conc = 50000, kappab =0,
                 ifAssignSpk=False,
                 A1 = np.array([1.5]), A2 = np.array([0]), tau1 = np.array([0.1]),
                 tau2 = np.array([0.3]), tauOn = np.array([0.01]), A1sigma = np.array([]), tau1sigma = np.array([]),
                 snr = np.array([5]), dffmax = np.array([93]), frameRate = 30, samplingRate = 1000,
                 recycleSpikeTimes = False, params=None):

        
        # initialize
        #--------------------#--------------------#--------------------#--------------------#--------------------#
        if params is None:
            self.params = modelCa2Params.ModelCa2Params(
                spk_recmode=spk_recmode,dur=dur, offset=offset, doPlot=doPlot, doVectorized=doVectorized, 
                cellNum=cellNum, spikeRate=spikeRate, ca_genmode=ca_genmode, ca_amp=ca_amp, ca_amp1=ca_amp1,
                ca_tau1=ca_tau1, ca_rest=ca_rest, ca_gamma=ca_gamma, ca_kappas=ca_kappas,
                ifAssignSpk=ifAssignSpk,
                ca_onsettau=ca_onsettau, kd=kd, conc=conc, kappab=kappab,A1=A1, A2=A2, tau1=tau1,
                tau2=tau2, tauOn=tauOn, A1sigma=A1sigma, tau1sigma=tau1sigma, snr=snr, dffmax=dffmax, 
                frameRate=frameRate, samplingRate=samplingRate, recycleSpikeTimes=False)
        else:
            self.params = params

        self.ifAssignSpk = self.params.ifAssignSpk
        self.cellNum = self.params.simulation['cellNum']
        self.x = np.arange(0, self.params.simulation['dur'], 1/self.params.simulation['samplingRate']) #30000
        self.xwithOffset = np.arange(0, self.params.simulation['dur']+self.params.simulation['offset'], #31000
                                     1/self.params.simulation['samplingRate'])
        self.ca = np.zeros((self.cellNum, len(self.xwithOffset)))
        self.fluor = np.zeros((self.cellNum, len(self.xwithOffset)))
        
        self.noisyFluor = np.zeros((self.cellNum, len(self.x)))
        
        self.xSampled = np.arange(0, self.params.simulation['dur'], 1/self.params.simulation['frameRate'])
        self.sampledFluor = np.zeros((self.cellNum, len(self.xSampled)))
        self.dff = np.zeros((self.cellNum, len(self.xSampled)))
        self.baseline = np.zeros(self.cellNum)
        

        # update
        #--------------------#--------------------#--------------------#--------------------#--------------------#
        # generate spkTimes for each neruon
        self.generatePossionSpkTrain()
        # spikes to calcium to fluorescence
        self.update_spk2ca2fluor_spkTimes()
        self.plot_traceWithSpikes()
        # calculated gaussian sd and add noise to fluorescence
        self.update_peakA_noiseSD()
        self.update_addNoise()
        # down-sampling high-frequency fluorescence
        self.update_downsampling()
        # de-baseline
        self.update_dffBaseline()

        # assign spike times
        #--------------------#--------------------#--------------------#--------------------#--------------------#
        self.spkTimes = self.params.simulation['spkTimes']




    def generatePossionSpkTrain(self, dt=0.0001):
    
        '''
        update spkTimes in parameter space using PoissonSpikeTrain() function in Functions.py
        Note that spkTimes are generated with offset, which combines with operations in update_spk2ca2fluor() to 
        make no spikes in offset period
        if not assign spk: generate; else: keep original
        '''
        
        if not self.ifAssignSpk:
            for i in range(self.cellNum):
                spkRate = self.params.simulation['spikeRate'][i]
                dur = self.params.simulation['dur'] + self.params.simulation['offset']
                self.params.simulation['spkTimes'][i] = PoissonSpikeTrain(spkRate, dur, dt)
                
        
        return(self)
    

    def update_spk2ca2fluor_spkTimes(self):
    
        '''
        update fluorecence for each neuron, both linear and saturation
        update ca only when ca_genmode='satDFF'
        update self.params.simulation['spkTimes'] for each neuron, remove any spikes in offset period
        also update self.x
        '''
        duration = self.params.simulation['dur'] + self.params.simulation['offset']

        for m in range(self.cellNum):
            currentSpkT = np.array(self.params.simulation['spkTimes'][m].copy())
            
            if len(currentSpkT) == 0:
                logging.warning('No spikes for cell',m, 'skipping')
                
                if m == 0:
                    raise Exception('Must have spikes for cell 1. Try to increase firing rate / simulation duration.')

            
            if len(self.params.simulation['A1sigma']) !=0 and len(self.params.simulation['tau1sigma']) !=0:
                # convolution for each spike (slower, allows variable model calcium transient)
                for n in range(len(currentSpkT)):
                    currentA1 = np.random.normal(self.params.simulation['A1'][m], 
                                                self.params.simulation['A1'][m]*self.params.simulation['A1sigma']) # generate a vec of normal distribution with mean=A1[m] std=~
                    currentTau1 = np.random.normal(self.params.simulation['tau1'][m],
                                                self.params.simulation['tau1'][m]*self.params.simulation['tau1sigma'])
                    y = spkTimes2Calcium(currentSpkT[n], self.params.simulation['tauOn'][m], currentA1,currentTau1,
                                        self.params.simulation['A2'][m], self.params.simulation['tau2'][m],
                                        self.params.simulation['samplingRate'], duration)
                    self.fluor[m,:] = self.fluor[m,:] + y[0:len(self.xwithOffset)]
            else:
                if self.params.simulation['ca_genmode'] == 'linDFF':

                    # convolution over all spikes (faster, same model calcium transient)
                    modelTransient = spkTimes2Calcium(0,self.params.simulation['tauOn'][m],
                                            self.params.simulation['A1'][m], self.params.simulation['tau1'][m], 
                                            self.params.simulation['A2'][m], self.params.simulation['tau2'][m], 
                                            self.params.simulation['samplingRate'], duration)
                    
                    if self.ifAssignSpk and len(self.params.simulation['spkTrain'])!=0:
                        spkVector = self.params.simulation['spkTrain'][m]
                    else:
                        spkVector = np.array([0]*len(self.xwithOffset))
                        for i in range(len(currentSpkT)):
                            # this is to get the most accurate spike times
                            # at which time point, the spike happens, then mark that index as value 1 in spkVector
                            idx = pd.Series(abs(currentSpkT[i]-self.xwithOffset)).idxmin() 
                            spkVector[idx] = spkVector[idx]+1

                    dffConv = np.convolve(spkVector, modelTransient) # convolution happens here
                    self.fluor[m,:] = dffConv[0:len(self.xwithOffset)] # update the m-th neuron dff

                elif self.params.simulation['ca_genmode'] == 'satDFF':
                    # taking saturation into account by solving the single comp model differential equation
                    # piecewise, then applying nonlinear transformation from ca to dff
                    self.ca[m,:] = spkTimes2FreeCalcium(currentSpkT, self.params.simulation['ca_amp'][m],
                                        self.params.simulation['ca_gamma'][m], self.params.simulation['ca_onsettau'][m],
                                        self.params.simulation['ca_rest'][m], self.params.simulation['ca_kappas'][m],
                                        self.params.simulation['kd'], self.params.simulation['conc'],
                                        self.params.simulation['samplingRate'], duration)
                    self.fluor[m,:] = Calcium2Fluor(self.ca[m,:], self.params.simulation['ca_rest'][m],
                                                self.params.simulation['kd'], self.params.simulation['dffmax'][m])


            currentSpkT = currentSpkT - self.params.simulation['offset']  # without first 1s offset
            currentSpkT = currentSpkT[currentSpkT>=0] # remove any spikes in offset period
            if not self.ifAssignSpk:
                self.params.simulation['spkTimes'][m] = currentSpkT # spikes without offset

        # update x
        xPlot = self.xwithOffset - self.params.simulation['offset']
        self.fluor = self.fluor[:,xPlot>=0]
        self.x = xPlot[xPlot>=0]

        return(self)    
                

    def update_peakA_noiseSD(self):
        '''
        update each neuron's
            - peakA: expected peak amplitude based on exponential rise and decay
            - noiseSD: how much Gaussian noise is added to the DFF trace
        '''

        if self.params.simulation['ca_genmode'] == 'linDFF':
            self.params.simulation['peakA'] = self.params.simulation['A1'] * (self.params.simulation['tau1']/self.params.simulation['tauOn']*(self.params.simulation['tau1']/self.params.simulation['tauOn']+1)**(-(self.params.simulation['tauOn']/self.params.simulation['tau1']+1)))
        
        elif self.params.simulation['ca_genmode'] == 'satDFF':
            self.params.simulation['ca_amp1'] = self.params.simulation['ca_amp']/(1+self.params.simulation['ca_kappas']+self.params.simulation['kappab'])
            self.params.simulation['ca_tau1'] = (1+self.params.simulation['ca_kappas']+self.params.simulation['kappab'])/self.params.simulation['ca_gamma']
            PeakCa = self.params.simulation['ca_amp1'] + self.params.simulation['ca_rest']
            PeakDFF = Calcium2Fluor(PeakCa,self.params.simulation['ca_rest'],self.params.simulation['kd'],self.params.simulation['dffmax'])
            self.params.simulation['peakA'] = PeakDFF * (self.params.simulation['ca_tau1']/self.params.simulation['ca_onsettau']*(self.params.simulation['ca_tau1']/self.params.simulation['ca_onsettau']+1)**(-(self.params.simulation['ca_onsettau']/self.params.simulation['ca_tau1']+1)))

        else:
            raise Exception('Calcium trace generation mode illdefined. Chose from linDFF, satDFF.')

        self.params.simulation['noiseSD'] = self.params.simulation['peakA']/self.params.simulation['snr'] 

        return(self)
    
    
    def update_addNoise(self):
        '''
        add gaussian noise to fluorescence
        the sd used for each neuron is calculated in update_peakA_noiseSD()
        self.noisydff got updated
        '''

        for m in range(self.cellNum):
            if len(self.params.simulation['spkTimes'][m]) == 0:
                continue
        
            whiteNoise = self.params.simulation['noiseSD'][m]*np.random.normal(size=len(self.x)) # e.g., in range [-5, 5]
            self.noisyFluor[m,:]= self.fluor[m,:] + whiteNoise

        return (self)


    def update_downsampling(self):
        '''
        the fps used to generate fluorescence is 1000
        however, the recording frame rate would be much lower, e.g., 30
        need to down sampling fluorescnece to generate dff much more like the one we got from recording machine
        self.xSampled and self.sampledDFF got updated
        '''

        self.xSampled =  np.arange(0, max(self.x), 1/self.params.simulation['frameRate'] ) - (0.5*1/self.params.simulation['frameRate'])
        _,idxList = findClosest(self.xSampled, self.x)
        
        for m in range(self.cellNum):
            if len(self.params.simulation['spkTimes'][m]) == 0:
                continue
            
            self.sampledFluor[m,:] = self.noisyFluor[m,idxList]

        
        return (self)


    def update_dffBaseline(self):
        '''
        calculate baseline for each neuron's downsampled noisydff
        minue baseline to get dff
        self.dff got updated
        '''
        for m in range(self.cellNum):
            if len(self.params.simulation['spkTimes'][m]) == 0:
                continue
            # now perform baseline setting based on ratio of expected and real SD of trace
            ratioSD = np.std(self.sampledFluor[m,:])/self.params.simulation['noiseSD'][m]
            
            # empirical function describing baseperc as function of ratioSD
            basePerc = 50 - 50/(1+np.exp(-(ratioSD-1-self.params.simulation['snr'][m]*0.25)*10/self.params.simulation['snr'][m]))

            baseValue = np.percentile(self.sampledFluor[m,:], basePerc)
            
            self.dff[m,:] = self.sampledFluor[m,:] - baseValue
            self.baseline[m] = baseValue

        return(self)
    
    
    def plot_traceWithSpikes(self, plotX=None, dfToPlot=None,ifPlot=False, ifSave=False, savePath=None, filename=None):
        '''
        plot one type trace with spikes marked using error bar
        can be dff/noisy/sampledNoisy dff, the plotX should be corresponding ones
        by default, this function plot self.dff, self.x
        '''
        if dfToPlot is None and plotX is None:
            dfToPlot = self.fluor
            plotX = self.x

        dffMax = dfToPlot[:].max().max()

        if ifPlot:
            # only 1 neuron
            if self.cellNum == 1:
                m=0
                currentDff = dfToPlot[m, :]
                
                plt.figure(figsize=(20, 2))
                plt.plot(plotX, currentDff, c='black')

                # plot spikes
                for n in range(len(self.params.simulation['spkTimes'][m])):
                    plt.errorbar(x=self.params.simulation['spkTimes'][m][n], y=max(currentDff)+2, yerr=1, c='blue', elinewidth=2)
                    
            elif self.cellNum > 1:
                fig, ax = plt.subplots(self.cellNum,1, figsize=(20,2*self.cellNum),facecolor='w', edgecolor='k')
                for m in range(self.cellNum):
                    label='neuron'+str(m)
                    currentDff = dfToPlot[m, :]
                    ax[m].plot(plotX, currentDff, c='black', label=label)
                    ax[m].legend(loc=1)   
                    
                    # plot spikes (using errorbar)
                    for n in range(len(self.params.simulation['spkTimes'][m])):
                        ax[m].errorbar(x=self.params.simulation['spkTimes'][m][n], y=max(currentDff)+2, yerr=1,c='blue', elinewidth=2)
            else:
                raise Exception('No neuron in this model')


            if ifSave:
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                plt.savefig(os.path.join(savePath, filename))
                plt.close()
            else:
                plt.show()    

    
    def plot_allTraces(self, ifSave=False, savePath=None, filename=None):
        '''
        plot dff, noisyDff, sampledDFF, spikes on one plot
        '''
        
        dffMax = self.noisyFluor[:].max().max()

        # only 1 neuron
        if self.cellNum == 1:
            m=0
            currentDff = self.sampledFluor[m, :]
            
            plt.figure(figsize=(20, 2))
            plt.plot(self.x, self.noisyFluor[m, :], c='black', alpha=0.2, label='noisyFluor')
            plt.plot(self.xSampled, self.sampledFluor[m, :], c='black', alpha=0.8, lw=2, label='noisyFluor downsampled')
            plt.plot(self.xSampled, self.dff[m, :], c='red', lw=0.4, label='dff')

            # plot spikes
            for n in range(len(self.params.simulation['spkTimes'][m])):
                if n==0:
                    plt.errorbar(x=self.params.simulation['spkTimes'][m][n], label='spikes',
                                    y=max(currentDff)+2, yerr=1, c='blue', elinewidth=2)
                else:
                    plt.errorbar(x=self.params.simulation['spkTimes'][m][n], 
                                    y=max(currentDff)+2, yerr=1, c='blue', elinewidth=2)
            
            plt.legend(loc=1)
                
        elif self.cellNum > 1:
            
            fig, ax = plt.subplots(self.cellNum,1, figsize=(20,2*self.cellNum),facecolor='w', edgecolor='k')
            for m in range(self.cellNum):
                label='neuron'+str(m)
                currentDff = self.sampledFluor[0, :]

                if m==0 or m==self.cellNum-1:
                    ax[m].plot(self.x, self.noisyFluor[m, :], c='black', alpha=0.2, lw=4, label='noisyFluor')
                    ax[m].plot(self.xSampled, self.sampledFluor[m, :], c='black', alpha=0.8, lw=2, label='noisyFluor downsampled')
                    ax[m].plot(self.xSampled, self.dff[m, :], c='red', lw=0.4, label='dff')
                else:
                    ax[m].plot(self.x, self.noisyFluor[m, :], c='black', alpha=0.2, lw=4, )
                    ax[m].plot(self.xSampled, self.sampledFluor[m, :], c='black', alpha=0.8, lw=2, )
                    ax[m].plot(self.xSampled, self.dff[m, :], c='red', lw=0.4, )

                ax[m].set_title(label)
                ax[m].legend(loc=1)   
                
                # plot spikes (using errorbar)
                for n in range(len(self.params.simulation['spkTimes'][m])):
                    if n==0 and (m==0 or m==self.cellNum-1):
                        ax[m].errorbar(x=self.params.simulation['spkTimes'][m][n], label='spikes',
                                    y=max(currentDff)+2, yerr=1, c='blue', elinewidth=2)
                    else:
                        ax[m].errorbar(x=self.params.simulation['spkTimes'][m][n], 
                                    y=max(currentDff)+2, yerr=1, c='blue', elinewidth=2)
                    ax[m].legend(loc=1)   
        else:
            raise Exception('No neuron in this model')

        
        plt.tight_layout()
        if ifSave:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(os.path.join(savePath, filename))
            plt.close()
        else:
            plt.show()    


  






