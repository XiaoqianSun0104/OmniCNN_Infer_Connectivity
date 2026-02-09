#!/usr/bin/env python
# peelReconstruction.py
#
# Author: Xiaoqian Sun, 07/2023 
#
# peel and reconstruction object


# Import Packages
#========================================================================================
import os 
import time
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid

import warnings
warnings.filterwarnings('ignore')

import peelParams
from peelingFunctions import *



# Class
#========================================================================================
class peelReconstr(object):
    '''
    peel and Reconstruction process
        - peel
        - reconstruction
    '''


    def __init__(self,dff,xSampled,spkTimes,maxiter=19999,
                lowerT=0.05,upperT=0.05,optimMethod='L-BFGS-B',verbose=False,ifPlot=False,optTiming=True,

                onsetposition=0.0,ca_genmode='linDFF',ca_onsettau=0.02,ca_amp=7600,ca_gamma=400,ca_amp1=0,ca_tau1=0,
                ca_kappas=100,ca_rest=50,ca_current=50,tauOn=0.01,offset=0,A1=1.5,tau1=0.1,A2=0,tau2=0.03,
                integral=0,scale=1.0,negintegral=0,

                frameRate=30,noiseSD=1.2,indicator='OGB-1',dffmax=93,kd=250,conc=50000,kappab=0,
                
                schmitt=[1.75, -1, 0.3],
                spk_recmode='linDFF',padding=20,
                smtthigh=0.4,smttlow=0.1,smttbox=3,smttmindur=0.3,smttnumevts=0,smttmindurFrames=3,smttlowMinEvents=1,
                slidwinsiz=10.0,maxbaseslope=0.5,evtfound=False,nextevt=0,nextevtframe=0,intcheckwin=0.1,intacclevel=0.4,
                fitonset=0,fitwinleft=0.5,fitwinright=0.5,negintwin=0.1,negintacc=0.2,stepback=0.2,fitupdatetime=0.5,
                optimizeSpikeTimes=False,doPlot=False,evtaccepted=False,

                params=None):

        # initialize
        #--------------------#--------------------#--------------------#--------------------#--------------------#
        if params is None:
            self.params = peelParams.peelParams(
                dff=dff,
                onsetposition=onsetposition,ca_genmode=ca_genmode, ca_onsettau=ca_onsettau, ca_amp=ca_amp,
                ca_gamma=ca_gamma,ca_amp1=ca_amp1,ca_tau1=ca_tau1, ca_kappas=ca_kappas,ca_rest=ca_rest,
                ca_current=ca_current,tauOn=tauOn,offset=offset,A1=A1,tau1=tau1,A2=A2,tau2=tau2,
                integral=integral,scale=scale,negintegral=negintegral,

                frameRate=frameRate,noiseSD=noiseSD,indicator=indicator,dffmax=dffmax,kd=kd,conc=conc,kappab=kappab,

                schmitt=schmitt,spk_recmode=spk_recmode,padding=padding,smtthigh=smtthigh,smttlow=smttlow,
                smttbox=smttbox,smttmindur=smttmindur,smttnumevts=smttnumevts,slidwinsiz=slidwinsiz,
                maxbaseslope=maxbaseslope,evtfound=evtfound,nextevt=nextevt,nextevtframe=nextevtframe,
                intcheckwin=intcheckwin,intacclevel=intacclevel,fitonset=fitonset,fitwinleft=fitwinleft,
                fitwinright=fitwinright,negintwin=negintwin,negintacc=negintacc,stepback=stepback,
                fitupdatetime=fitupdatetime,optimizeSpikeTimes=optimizeSpikeTimes,doPlot=doPlot,
                smttmindurFrames=smttmindurFrames,smttlowMinEvents=smttlowMinEvents,evtaccepted=evtaccepted)
        else:
            self.params = params

        self.nexttim = 1/self.params.exp_p['frameRate']
        self.lowerT=lowerT
        self.upperT=upperT 
        self.xSampled = xSampled
        self.optimMethod=optimMethod
          
        self.dffPeel = dff
        self.dffRecon = np.ones(len(dff))
        self.spkTimes = spkTimes # ground truth
        self.spkTimesRecon = [] # predict ones in peeling()
        
        self.maxiter = maxiter
        self.verbose = verbose
        self.ifPlot=ifPlot

             
        

        # update
        #--------------------#--------------------#--------------------#--------------------#--------------------#
        self.wrap_SingleFluorTransient(self.params.ca_p['ca_genmode'], 1/self.params.exp_p['frameRate'])
        self.update_smttParams()
        # peel
        self.update_Peeling(verbose=self.verbose, ifPlot=self.ifPlot)
        self.update_optimizeTiming(self.optimMethod)
        self.peel = self.params.data['peel']
        self.spkTimesRecon = self.params.data['spikes']
        # reconstruction
        self.update_dffRecon()




    def wrap_SingleFluorTransient(self, mode, starttim):

        self.params.ca_p,self.params.exp_p,self.params.data=SingleFluorTransient(
                        self.params.ca_p, self.params.exp_p, self.params.data, mode, starttim)
        
        return(self)
    
    
    def wrap_FindNextEvent(self, nexttim, ifPlot=False, verbose=False):
        '''
        calling function FindNextEvent() in Functions.py
        wrap it, since this function will be called several times
        '''
        self.params.ca_p,self.params.peel_p,self.params.data = FindNextEvent(self.params.ca_p, self.params.exp_p, 
                                                                self.params.peel_p, self.params.data, nexttim,
                                                                ifPlot=False,verbose=False)
        return(self)


    def update_smttParams(self):
        self.params.peel_p['smttmindurFrames'] = math.ceil(self.params.peel_p['smttmindur']*self.params.exp_p['frameRate'])
        self.params.peel_p['smttlowMinEvents'] = 1
        self.nexttim = 0 #1/self.params.exp_p['frameRate']

        return (self)
    
    
    def update_Peeling(self,verbose=False,ifPlot=False):
        '''
        main process.
        for more details, refer to Peeling() function in Functions.py
        '''

        self.wrap_FindNextEvent(self.nexttim,verbose=verbose,ifPlot=ifPlot)
        if self.params.peel_p['evtfound']:
            self.params.data['numspikes'] = self.params.data['numspikes'] + 1
            self.params.data['spikes'][self.params.data['numspikes']] = self.params.peel_p['nextevt']
            self.wrap_SingleFluorTransient(self.params.peel_p['spk_recmode'], self.params.peel_p['nextevt'])
            self.params.data['model'] = self.params.data['model'] + self.params.data['singleTransient']
            
            # if verbose:
            #     print('first pike before while loop --------------------')
            #     print("self.params.data['numspikes'] =", self.params.data['numspikes'], "self.params.peel_p['nextevt'] =", self.params.peel_p['nextevt'])
            #     plt.figure(figsize=(15, 1))
            #     plt.plot(self.params.data['singleTransient'], c='r', lw=3);plt.title('singleTransient'); plt.show()
            #     plt.figure(figsize=(15, 1))
            #     plt.plot(self.params.data['model'], c='b',lw=3); plt.title('self.params.data["model"]');plt.show()
        
        
        # now data['numspikes']=1
        # while loop to exam 1st event and find remaining events
        iteration = 0
        nexttimMem = float('inf')
        nexttimCounter = 0
        timeStepForward = 2/self.params.exp_p['frameRate']
        while self.params.peel_p['evtfound']:
        
            # check integral after subtracting Ca transient
            if self.params.peel_p['spk_recmode'] == 'linDFF':
                pass
            elif self.params.peel_p['spk_recmode'] == 'satDFF':
                self.params.ca_p['onsetposition'] = self.params.peel_p['nextevt']
                self.params.ca_p = IntegralofCaTransient(self.params.ca_p, self.params.peel_p, 
                                                         self.params.exp_p, self.params.data)
            
            # remove the first event
            dummy = self.params.data['peel'] - self.params.data['singleTransient'] 
            # exam the previous spike valid or not
            # in 1st run, exam the spike before the while loop
            spk_timDiff = list(abs(self.params.data['tim'] - self.params.data['spikes'][self.params.data['numspikes']]))
            startIdx = spk_timDiff.index(min(spk_timDiff))
            spk_checkTimDiff = list(abs( self.params.data['tim'] - (self.params.data['spikes'][self.params.data['numspikes']]+self.params.peel_p['intcheckwin'])))
            stopIdx = spk_checkTimDiff.index(min(spk_checkTimDiff))

            if verbose:
                print('-----------------------------------------------------------------------------')
                print('spikeTime at', self.params.data['spikes'][self.params.data['numspikes']], 
                      'spike starts at', startIdx, ', and check window ends at', stopIdx, end=' ')
                # plot currentTransient to get a sense if an event exists
                plt.figure(figsize=(15,2))
                plt.plot(self.params.data['dff'], label='dff', color='grey', alpha=0.2, lw=4)
                plt.plot(dummy, label='residual dummy', c='blue', alpha=0.4, lw=2)
                plt.plot(self.params.data['peel'], label='peel', color='orange', alpha=1, lw=1)
                plt.title('whole dummy'); plt.legend(loc=1); plt.show()

                # only plot partial 
                sIdx = max(0, startIdx); eIdx = min(len(self.params.data['dff']), stopIdx)
                plt.figure(figsize=(15,2))
                plt.plot(self.params.data['dff'][sIdx:eIdx], label='dff', color='grey', alpha=0.2, lw=4)
                plt.plot(dummy[sIdx:eIdx], label='residual dummy', c='blue', alpha=0.4, lw=2)
                plt.plot(self.params.data['peel'][sIdx:eIdx], label='peel', color='orange', alpha=1, lw=1)
                plt.grid(axis='y', linewidth=1.5, linestyle='--', color='gray')
                plt.title('partial dummy from f'+str(sIdx)+' to f'+str(eIdx)); plt.legend(loc=1); plt.show()

                print('dummy[startIdx:stopIdx]:', dummy[startIdx:stopIdx])


            # clean up indices
            if startIdx < stopIdx:
                currentTim = self.params.data['tim'][startIdx:stopIdx]
                currentPeel = dummy[startIdx:stopIdx]
                currentIntegral = trapezoid(currentPeel, x=currentTim)# integral after one peel
            else:
                # if enters here, startIdx is the last data point and we should not accept it as a spike
                currentIntegral = self.params.ca_p['negintegral']*self.params.peel_p['negintacc']

            
            if currentIntegral > (self.params.ca_p['negintegral']*self.params.peel_p['negintacc']):   
                '''
                this if condition means only accept this spike if removing it doesn't leave too big of a hole in the trace
                    - if subtracting singleTransient from the trace leaves a large “hole” — a big negative dip
                    - the integral of this dummy region becomes strongly negative
                    - then it's a red flag: the candidate event wasn’t justified
                    - which goes to 'else' part
                '''
                if verbose:
                    print('currentIntegral = ', currentIntegral, '>', self.params.ca_p['negintegral']*self.params.peel_p['negintacc'])
                    print('dummy_checkWin Integral>negIntegral, real spike, peel off singleTransient')

                self.params.data['peel'] = self.params.data['peel'] - self.params.data['singleTransient'] # peel off happens here
                self.nexttim = self.params.data['spikes'][self.params.data['numspikes']] - self.params.peel_p['stepback']
                if self.nexttim < 0:
                    self.nexttim = 0 #1/self.params.exp_p['frameRate'] 
                if verbose:
                    print('step back to nexttim='+str(round(self.nexttim,4))+' and check again.')
            else:
                if verbose:
                    print('dummy_checkWin Integral<=negIntegral. Revoke spike, numspikes, and singleTransient in data.model')
                
                self.params.data['spikes'][self.params.data['numspikes']] = 0
                self.params.data['numspikes'] = self.params.data['numspikes']-1
                self.params.data['model'] = self.params.data['model'] - self.params.data['singleTransient']
                self.nexttim = self.params.peel_p['nextevt'] + timeStepForward

                if verbose:
                    print('step forward to nexttim='+str(round(self.nexttim,4))+' and check event')
            
            if verbose:
                plt.figure(figsize=(15,1))
                plt.plot(self.params.data['model'], label='model')
                plt.title('current model')
                plt.legend(loc=1)
                plt.show()

            # find next event
            self.params.peel_p['evtaccepted'] = False
            self.wrap_FindNextEvent(self.nexttim,verbose=verbose,ifPlot=ifPlot)
            if self.params.peel_p['evtfound']:
                self.params.data['numspikes'] = self.params.data['numspikes'] + 1
                self.params.data['spikes'][self.params.data['numspikes']] = self.params.peel_p['nextevt']
                self.wrap_SingleFluorTransient(self.params.peel_p['spk_recmode'], self.params.peel_p['nextevt'])
                
                self.params.data['model'] = self.params.data['model'] + self.params.data['singleTransient']
            else:
                break


            iteration = iteration+1
            if self.nexttim == nexttimMem:
                nexttimCounter = nexttimCounter + 1
            else:
                nexttimMem = self.nexttim
                nexttimCounter = 0
            
            if nexttimCounter > 50:
                self.nexttim = self.nexttim + timeStepForward 
            
            if iteration > self.maxiter :
                logging.warning('Reached maxiter (#1.0f). nexttim=#1.2f. Timeout!',self.maxiter ,self.nexttim)
                break


        #print("self.params.data['spikes'][0:5]:", self.params.data['spikes'][0:5])
        if len(self.params.data['spikes']) > self.params.data['numspikes']:
            self.params.data['spikes'] = self.params.data['spikes'][1:self.params.data['numspikes']+1]   
            # remove any extra 0
            self.params.data['spikes'] = self.params.data['spikes'][self.params.data['spikes'] != 0]
            #print("gets in to if, now self.params.data['spikes'][0:5]:", self.params.data['spikes'][0:5])

        return(self)


    # def update_optimizeTiming(self, optimMethod):
    #     '''
    #     optimization of reconstructed spike times to improve timing
    #     data['modelConvolve'] got updated. Before we used data['model'] which will change the transientShape at each peak
    #     '''

    #     optMethod = optimMethod
    #     if len(self.params.data['spikes']) and self.params.peel_p['optimizeSpikeTimes']:
    #         if self.params.peel_p['spk_recmode'] == 'linDFF':
    #             spikes,_ = PeelingOptimizeSpikeTimes(self.params.data['dff'], self.params.data['spikes'], 
    #                                                  self.lowerT, self.upperT,
    #                                                 self.params.exp_p['frameRate'], self.params.ca_p['tauOn'], 
    #                                                 self.params.ca_p['A1'], self.params.ca_p['tau1'], optMethod)
    #         elif self.params.peel_p['spk_recmode'] == 'satDFF':
    #             spikes,_ = PeelingOptimizeSpikeTimesSaturation(self.params.data['dff'], self.params.data['spikes'], 
    #                                     self.lowerT, self.upperT, self.params.ca_p['ca_amp'], 
    #                                     self.params.ca_p['ca_gamma'], self.params.ca_p['ca_onsettau'], 
    #                                     self.params.ca_p['ca_rest'], self.params.ca_p['ca_kappas'], 
    #                                     self.params.exp_p['kd'], self.params.exp_p['conc'], 
    #                                     self.params.exp_p['dffmax'], self.params.exp_p['frameRate'],
    #                                     len(self.params.data['dff'])/self.params.exp_p['frameRate'], optMethod)
    #         else:
    #             raise Exception('Undefined mode')

    #         self.params.data['spikes'] = spikes

        
    #     # loop to create spike train vector from spike times
    #     self.params.data['spiketrain'] = np.zeros(len(self.params.data['dff']))
    #     for i in range(len(self.params.data['spikes'])):
    #         spk_timDiff = list(abs(self.params.data['spikes'][i]-self.params.data['tim']))
    #         idx = spk_timDiff.index(min(spk_timDiff))
    #         self.params.data['spiketrain'][idx] = self.params.data['spiketrain'][idx]+1

    #     # re-derive model and residuals after optimization
    #     if self.params.peel_p['spk_recmode'] == 'linDFF':
    #         modelTransient = spkTimes2Calcium(0, self.params.ca_p['tauOn'], self.params.ca_p['A1'], 
    #                                 self.params.ca_p['tau1'], self.params.ca_p['A2'], self.params.ca_p['tau2'],
    #                             self.params.exp_p['frameRate'], len(self.params.data['dff'])/self.params.exp_p['frameRate'] )
    #         #self.params.data['model'] = np.convolve(self.params.data['spiketrain'], modelTransient)
    #         #self.params.data['model'] = self.params.data['model'][0:len(self.params.data['tim'])]
    #         self.params.data['modelConvolve'] = np.convolve(self.params.data['spiketrain'], modelTransient)
    #         self.params.data['modelConvolve'] = self.params.data['modelConvolve'][0:len(self.params.data['tim'])]
    #     elif self.params.peel_p['spk_recmode'] == 'satDFF':
    #         modeltmp = spkTimes2FreeCalcium(self.params.data['spikes'], self.params.ca_p['ca_amp'], self.params.ca_p['ca_gamma'], 
    #                             self.params.ca_p['ca_onsettau'], self.params.ca_p['ca_rest'], self.params.ca_p['ca_kappas'],
    #                             self.params.exp_p['kd'], self.params.exp_p['conc'], self.params.exp_p['frameRate'], len(self.params.data['dff'])/self.params.exp_p['frameRate'] )
    #         #self.params.data['model'] = Calcium2Fluor(modeltmp, self.params.ca_p['ca_rest'], self.params.exp_p['kd'], self.params.exp_p['dffmax'])
    #         self.params.data['modelConvolve'] = Calcium2Fluor(modeltmp, self.params.ca_p['ca_rest'], self.params.exp_p['kd'], self.params.exp_p['dffmax'])

    #     self.params.data['peel'] = self.params.data['dff'] - self.params.data['modelConvolve']


    #     return (self)

  
    def update_optimizeTiming(self, optimMethod, batch_size=100):
        '''
        a faster version - chatgpt
        '''

        optMethod = optimMethod
        spikes_in = self.params.data['spikes']
        if len(spikes_in) == 0 or not self.params.peel_p['optimizeSpikeTimes']:
            return self
        
        # === Perform optimization in batches ===
        spikes_out = []
        startTime=time.time()
        if self.params.peel_p['spk_recmode'] == 'linDFF':
            for i in range(0, len(spikes_in), batch_size):
                batch = spikes_in[i:i + batch_size]
                if len(batch) < 2:
                    spikes_out += list(batch)
                    continue
                batch_out, _ = PeelingOptimizeSpikeTimes( self.params.data['dff'], batch, self.lowerT, self.upperT,
                                                         self.params.exp_p['frameRate'],self.params.ca_p['tauOn'], 
                                                         self.params.ca_p['A1'], self.params.ca_p['tau1'],optMethod)
                # endTime = time.time(); print('processing 1 batch cost', round((endTime-startTime)/60, 3))
                spikes_out += list(batch_out)
                # break

        elif self.params.peel_p['spk_recmode'] == 'satDFF':
            # Full optimization at once for satDFF
            spikes_out, _ = PeelingOptimizeSpikeTimesSaturation(self.params.data['dff'], spikes_in, self.lowerT, self.upperT,
                                                                self.params.ca_p['ca_amp'],self.params.ca_p['ca_gamma'],self.params.ca_p['ca_onsettau'],
                                                                self.params.ca_p['ca_rest'], self.params.ca_p['ca_kappas'], self.params.exp_p['kd'],
                                                                self.params.exp_p['conc'], self.params.exp_p['dffmax'], self.params.exp_p['frameRate'],
                                                                len(self.params.data['dff']) / self.params.exp_p['frameRate'], optMethod)
        else:
            raise Exception('Undefined spk_recmode.')
    
        # === Save optimized spikes ===
        spikes_out = np.sort(spikes_out)
        self.params.data['spikes'] = spikes_out
        
        # === Convert spike times to spike train ===
        tvec = self.params.data['tim']
        idxs = np.searchsorted(tvec, spikes_out)
        idxs = idxs[idxs < len(tvec)]
        spiketrain = np.bincount(idxs, minlength=len(tvec))
        self.params.data['spiketrain'] = spiketrain

        # === Reconstruct model signal ===
        if self.params.peel_p['spk_recmode'] == 'linDFF':
            modelTransient = spkTimes2Calcium(0, self.params.ca_p['tauOn'], self.params.ca_p['A1'], 
                                    self.params.ca_p['tau1'], self.params.ca_p['A2'], self.params.ca_p['tau2'],
                                self.params.exp_p['frameRate'], len(self.params.data['dff'])/self.params.exp_p['frameRate'] )
            #self.params.data['model'] = np.convolve(self.params.data['spiketrain'], modelTransient)
            #self.params.data['model'] = self.params.data['model'][0:len(self.params.data['tim'])]
            self.params.data['modelConvolve'] = np.convolve(self.params.data['spiketrain'], modelTransient)
            self.params.data['modelConvolve'] = self.params.data['modelConvolve'][0:len(self.params.data['tim'])]
        elif self.params.peel_p['spk_recmode'] == 'satDFF':
            modeltmp = spkTimes2FreeCalcium(self.params.data['spikes'], self.params.ca_p['ca_amp'], self.params.ca_p['ca_gamma'], 
                                self.params.ca_p['ca_onsettau'], self.params.ca_p['ca_rest'], self.params.ca_p['ca_kappas'],
                                self.params.exp_p['kd'], self.params.exp_p['conc'], self.params.exp_p['frameRate'], len(self.params.data['dff'])/self.params.exp_p['frameRate'] )
            #self.params.data['model'] = Calcium2Fluor(modeltmp, self.params.ca_p['ca_rest'], self.params.exp_p['kd'], self.params.exp_p['dffmax'])
            self.params.data['modelConvolve'] = Calcium2Fluor(modeltmp, self.params.ca_p['ca_rest'], self.params.exp_p['kd'], self.params.exp_p['dffmax'])

        self.params.data['peel'] = self.params.data['dff'] - self.params.data['modelConvolve']


        return (self)

    
    def update_dffRecon(self):   

        '''
        using predict spikes to reconstruct noisy dff
        self.dffRecon got updated
        '''

        dur = len(self.params.data['dff'])/self.params.exp_p['frameRate']

        if len(self.spkTimesRecon) != 0:

            if self.params.peel_p['spk_recmode'] == 'linDFF':
                modelTransient = spkTimes2Calcium(0, self.params.ca_p['tauOn'], self.params.ca_p['A1'], 
                            self.params.ca_p['tau1'], self.params.ca_p['A2'], self.params.ca_p['tau2'],
                            self.params.exp_p['frameRate'], dur)
                spkVector = np.zeros(len(self.xSampled))
                
                for i in range(len(self.spkTimesRecon)):
                    spkTimeR_diff = list(abs(self.spkTimesRecon[i]-self.xSampled))
                    idx = spkTimeR_diff.index(min(spkTimeR_diff))
                    spkVector[idx] = spkVector[idx]+1
                dffConv = np.convolve(spkVector, modelTransient)
                self.dffRecon = dffConv[0:len(self.xSampled)]
            
            elif self.params.peel_p['spk_recmode'] =='satDFF':
                ca = spkTimes2FreeCalcium(self.spkTimesRecon,self.params.ca_p['ca_amp'],self.params.ca_p['ca_gamma'],
                                        self.params.ca_p['ca_onsettau'],self.params.ca_p['ca_rest'],
                                        self.params.ca_p['ca_kappas'],self.params.exp_p['kd'], self.params.exp_p['conc'],
                                        self.params.exp_p['frameRate'],dur)
                dfftmp = Calcium2Fluor(ca,self.params.ca_p['ca_rest'],self.params.exp_p['kd'],self.params.exp_p['dffmax'])
                selfdffRecon = dfftmp[0:len(self.xSampled)]
            else:
                raise Exception('Model trace generation failed. Undefined SpkReconMode.')
            

        return (self)


        













