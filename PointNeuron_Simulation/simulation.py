'''
# simulation.py
# Author: Xiaoqian Sun, 07/03/2024
# Run simulation object and update dynamics in neuron/connectivity objects
'''


# Import Packages
import os 
import math
import time
import copy
import random
import logging
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import *
import neuron, connectivity
# from Simulation.Version1 import neuron, connectivity

import warnings
warnings.filterwarnings('ignore')



class Simulator(connectivity.Connectivity):
    
    
    def __init__(self,
                 T, dt, N, Ne, Ni, maxns, neurons, 
                 CM=None,cols=None,
                 p_exc2exc=0.2, p_exc2inh=0.2, p_inh2exc=0.2, p_inh2inh=0.2, 
                 we2e_max=1.5, we2i_max=1.5, wi2e_max=3, wi2i_max=3,
                 w_exc2exc=0.1, w_exc2inh=0.1, w_inh2exc=0.1, w_inh2inh=0.1,
                 **kwargs
                 ):
        # T/dt needs to be set first because the excuting order is:
        # set T/dt -- super.init, invoke update_attr(), then invoke update_attr() in super, then update Nt 
        # that is to say, you rewrite update_attr():super(), then update_arrt() in super.__init__() also changes
        self.T = T; self.dt = dt; self.maxns=maxns; self.neurons = neurons; self.Nt = int(self.T/self.dt)
        
        super().__init__(N, Ne, Ni, CM, cols,
                         p_exc2exc, p_exc2inh, p_inh2exc, p_inh2inh, 
                         we2e_max, we2i_max, wi2e_max, wi2i_max,
                         w_exc2exc, w_exc2inh, w_inh2exc, w_inh2inh, **kwargs)
        self = self.update_attr()
     
    def update_attr(self):
        super().update_attr()

        # incase T/dt change
        self.Nt = int(self.T/self.dt)

        return(self)


    ######################################################################################
                                #### Notes are written here ####
    ######################################################################################
    def run(self, ifVerbose=False, pickN=None, when2CalExp=None):
        startTime = time.time()

        # create some momentary storage lists
        netSpk = np.zeros((2, self.maxns))
        sb = np.zeros(self.N)        # to record whether a neuron fires or not in each t (sb-spike boolean)
        x = np.zeros(self.N)         # store momentary synTrace value for each neuron
        v = self.assign_initialV()   # stroe momentary membranePotential
        gE = np.zeros(self.N); gI = np.zeros(self.N)   # momentary gE/gI 
        refstate = np.zeros(self.N, dtype=int)         # refractory period
        CMW_list = []; CMW_list.append(self.CMW.copy())# record weight changing
        
        # save mean weights over time
        save_interval = int(100/self.dt)
        CMWM_temp_list = []; CMWM_temp_list.append(self.CMW.copy()) # store weith between current and next saving interval



        # if check firing rate shape
        if type(when2CalExp) != type(None):
            RMSEs = -1*np.ones(self.Nt-when2CalExp)
        if ifVerbose:
            print('at time=0, neuron'+str(pickN)+' starts from V_init='+str(v[pickN]))

        it=1; ns=0
        while it<self.Nt and ns<self.maxns:
            #---------------get neurons ready-----------------------------------------------------------
            for m in range(self.N):
                neuronObj = self.neurons[m]

                # if neurons have pre-assigned-spkTrains, they don't need simulation
                # they're only providing spk-input to other neurons
                # if not True: if not have pre-assigned-spkTrains
                if not neuronObj.get('ifAssignSpkTrain'): 
                    pre_exc, pre_inh = connectivity.get_connection_ExcInh(self.incomingCs, m, self.Ne)
                    
                    # if it<50:
                    #     print('pre_exc:', pre_exc, 'for m=', m)
                    #     print('pre_inh:', pre_inh, 'for m=', m)
                    #     print('it=', it, 'gE[m]=', gE[m])
                    #     print('it=', it, 'gI[m]=', gI[m])
                    # synaptic trace decay & record
                    x[m] = x[m] - (self.dt/neuronObj.paras['tau_stdp']) * x[m]
                    neuronObj.dynamics['synTrace'][it] = x[m]

                    # calculate gE/gI using w(t-1) & record
                    # note that we used CMW[pre_exc, m] here because m now is the postsynaptic neuron
                    # we want to get exc presynaptic neurons connect to m
                    # this dynamics aligns with static synapse:https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial3.html#section-1-1-simulate-synaptic-conductance-dynamics
                    gE[m] = gE[m] - (self.dt/neuronObj.paras['tau_excSyn'])*gE[m] + (neuronObj.paras['gE_bar']*self.CMW[pre_exc, m]*sb[pre_exc]).sum()
                    gE[m] = max(1e-5, gE[m]) # lower bound

                    gI[m] = gI[m] - (self.dt/neuronObj.paras['tau_inhSyn'])*gI[m] + (neuronObj.paras['gI_bar']*self.CMW[pre_inh, m]*sb[pre_inh]).sum()
                    gI[m] = max(1e-5, gI[m]) # lower bound

                    neuronObj.dynamics['gE'][it] = gE[m]; neuronObj.dynamics['gI'][it] = gI[m]



            #---------------simulation for each neuron i at time it---------------------------------------
            sb = np.zeros(self.N)
            for i in range(self.N):
                neuronObj = self.neurons[i]

                if not neuronObj.get('ifAssignSpkTrain'):
                    pre_exc, pre_inh = connectivity.get_connection_ExcInh(self.incomingCs, i, self.Ne)  # i's presynaptic exc/inh neurons
                    post_exc, post_inh = connectivity.get_connection_ExcInh(self.outgoingCs, i, self.Ne)# i's postsynaptic exc/inh neurons

                    #---------------update v---------------------------------------
                    if refstate[i] <= 0: # free from refactory period 
                        LeakC = -(v[i]-neuronObj.paras['V_LeakReversal'])
                        Ecurr = -(gE[i]*(v[i]-neuronObj.paras['V_excReversal']))/neuronObj.paras['g_Leak']
                        Icurr = -(gI[i]*(v[i]-neuronObj.paras['V_inhReversal']))/neuronObj.paras['g_Leak']

                        if type(neuronObj.paras['externalInput']) == np.ndarray:
                            Xcurr = neuronObj.paras['externalInput'][it]/neuronObj.paras['g_Leak']
                        elif type(neuronObj.paras['externalInput']) in [int, float, np.float64, np.int64]:
                            Xcurr = neuronObj.paras['externalInput']/neuronObj.paras['g_Leak']
                        else:
                            Xcurr=0

                        dv = (LeakC + Ecurr + Icurr + Xcurr) * (self.dt/neuronObj.paras['tau_m'])
                        v[i] = v[i] + dv

                        # lower bound
                        v[i] = max(v[i], neuronObj.paras['V_lowerbound'])

                    else: # if this cell is still in refactory period
                        LeakC=0; Ecurr=0; Icurr=0; Xcurr=0
                        if refstate[i] > 1:
                            v[i] = neuronObj.paras['V_fireThreshold']
                        else:
                            v[i] = neuronObj.paras['V_reset']
                        # count down refactory period
                        refstate[i] -= 1

                    #---------------check if a spike occurs---------------------------
                    if v[i] >= neuronObj.paras['V_fireThreshold'] and refstate[i] <= 0 and ns < self.maxns:

                        refstate[i] = neuronObj.paras['Ntref'] # reset refactory period
                        v[i] = neuronObj.paras['V_fireThreshold'] + 5 # this +5 is to make spike more prominent to see
                        neuronObj.dynamics['spkTrain'][it] = 1
                        neuronObj.dynamics['spkTimes'].append(it * self.dt) 
                        netSpk[0, ns] = it * self.dt
                        netSpk[1, ns] = i
                        sb[i] = 1     # mark this neuron firing at this time step
                        ns += 1

                        # update synaptic trace & record
                        x[i] = x[i] + 1
                        neuronObj.dynamics['synTrace'][it] = x[i]
                        
                        # update presynaptic weight to i (now i is postsynaptic neuron)
                        # if exc->i, i(post) fired, 𝑊_𝑖𝑗 → 𝑊_𝑖𝑗 + 𝜁𝑥_𝑗                     
                        self.CMW[pre_exc, i] += neuronObj.paras['lr_stdp_postSyn']*x[pre_exc]
                        # if inh->i, i(post) fired, 𝑊_𝑖𝑗 → 𝑊_𝑖𝑗 + 𝜂𝑥_𝑗 (inhibition not strong enough)    
                        self.CMW[pre_inh, i] += neuronObj.paras['lr_istdp']*x[pre_inh]
                        
                        # update postsynaptic weight from i (now i is the presynaptic neuron)
                        if neuronObj.neuronType==0: #e-e, e-i
                            # i(pre, exc) fired, 𝑊_𝑖𝑗 → 𝑊_𝑖𝑗 − 𝛾𝑥_𝑖 (note that 𝑥_𝑖 referes to postsynaptic neuron synaptic trace)
                            # i (pre) fire before post, 𝑊_𝑖𝑗 decreases
                            self.CMW[i, post_exc] -= neuronObj.paras['lr_stdp_preSyn']*x[post_exc]
                            self.CMW[i, post_inh] -= neuronObj.paras['lr_stdp_preSyn']*x[post_inh]
                        else: #i-e, i-i
                            # i(pre, inh) fired, 𝑊_𝑖𝑗 → 𝑊_𝑖𝑗 + 𝜂(𝑥_𝑖−𝛼)
                            # if fire close enough(x_i > 𝛼), inhibition is not strong enough, potentiate 𝑊_𝑖𝑗
                            self.CMW[i, post_exc] += neuronObj.paras['lr_istdp']*(x[post_exc]-neuronObj.paras['lr_istdp_window'])
                            self.CMW[i, post_inh] += neuronObj.paras['lr_istdp']*(x[post_inh]-neuronObj.paras['lr_istdp_window'])
                                        
                    #---------------record--------------------------------------------
                    neuronObj.dynamics['memPotential'][it]=v[i]
                    neuronObj.dynamics['EPSP'][it]=Ecurr*neuronObj.paras['g_Leak']
                    neuronObj.dynamics['IPSP'][it]=Icurr*neuronObj.paras['g_Leak'] 
                    neuronObj.dynamics['Xcurr'][it]=Xcurr*neuronObj.paras['g_Leak']
                    neuronObj.dynamics['LeakC'][it]=LeakC*neuronObj.paras['g_Leak']
                    

                    #if i==pickN and it%20==0 and ifVerbose:
                    if i==pickN and ifVerbose:
                        print('at time='+str(it))
                        print('  gE='+str(round(gE[i],3)), '| gI='+str(round(gI[i],3)))
                        print('  EPSP='+str(round(Ecurr,3)) ,'| IPSP='+str(round(Icurr,3)), '| Xcurr='+str(Xcurr), '| LeakC='+str(round(LeakC,3)))
                        print('  dv = '+str(round(dv,3)), '| v_it = '+str(v[i]))
                        print()
                    
                
                else: # a neuron has pre-assign-spkTrain
                    x[i] = neuronObj.get('assignSynTrace')[it]
                    sb[i] = neuronObj.get('assignSpkTrain')[it]

            
            
            # after a whole iteration, check upper bounds and save weight matrix
            self.CMW = connectivity.CMW_upperBound(self.CMW, self.Ne, 
                                      self.CMWParas['we2e_max'], self.CMWParas['we2i_max'], 
                                      self.CMWParas['wi2e_max'], self.CMWParas['wi2i_max'])
            
            # disable this for now, storing CMW at each it would explode array memory error
            #CMW_list.append(self.CMW.copy())
            if it%save_interval == 0:
                if CMWM_temp_list:       # make sure the list is not empty
                    mean_CMW = np.mean(np.stack(CMWM_temp_list), axis=0)
                    CMW_list.append(mean_CMW.copy())
                    CMWM_temp_list.clear()  # reset to store next # wegihts
            else:
                CMWM_temp_list.append(self.CMW.copy())


            it+=1


        # clean up any unfilled self.netSpk spaces
        netSpk = netSpk[:, 0:ns]
        endTime = time.time()
        if ifVerbose:
            print('The simulation takes', round((endTime-startTime)/60, 3), 'minutes')
        
        # assign
        self.ns = ns
        self.CMWs=CMW_list
        self.netSpk=netSpk
        self.CMW_df = connectivity.M_to_df(self.CMW, self.Ne, self.Ni, dataType=float)

        return(self)



    
    def assign_initialV(self):
        'get V_init for each neuron and put in an array'
        initialV = []
        for i in range(self.N):
            initV = self.neurons[i].paras['V_init']
            initialV.append(initV)

        return(np.array(initialV))
 
    def get_keysValues(self):

        '''
        return all values stored in the object
            either direct attributes (obj.simuLength)
            or dicts in the object (Obj.neuronPara)
        '''
        return (self.__dict__)
    
    def get_keys(self):
        '''
        return a list of attributes name, variables that can be accessed from the object
        '''
        return (self.__dict__.keys())
    
    def get_neuronKeys(self):
        return(self.neuorns[0].get_keys())
    
    def set(self, params_dict):
        """ 
        Set key-value pairs. E.g.,
            params_dict = {'V_excRevsal':16,  # in neuron.neuronPara{}
                           'simuLength':6}    # neuron.simuLength, direct attribute
            neuronObj.set(params_dict) to change corresponding values
        """
        for k, v in params_dict.items():
    
            if not hasattr(self, k):  # inside a {}
                attrNotExist_Flag = True
                
                # loop through subdic, locate key and change value
                for gr in list(self.__dict__.keys()):
                        d = getattr(self, gr)
                        if type(d) == dict:
                            if k in d:
                                d[k] = v
                                attrNotExist_Flag=False
                # if attrNotExist_Flag :
                #     logging.warning('No key in Simulator Object named {0}'.format(k))
            else:
                setattr(self, k, v)

        # if anything affect neurons
        for neuronObj in self.neurons:
            neuronObj.set(params_dict)

        self.update_attr()

    def get(self, key):
        """ 
        Get a value for a given key, which can either a direct attribute, or a key inside a dict (neuronPara)
        Raises an exception if no such group/key combination exists.
        """

        if not hasattr(self, key):  # inside a {}
            attrNotExist_Flag = True
            
            # loop through subdic, and get value
            for gr in list(self.__dict__.keys()):
                d = getattr(self, gr)
                if type(d) == dict:
                    if key in d:
                        attrNotExist_Flag=False
                        return (d[key])
            if attrNotExist_Flag:
                logging.warning('No key in Simulator Object named {0}'.format(key))
        else:
            return(getattr(self, key))

    def copy(self):
        # return fully independent copy of an object
        return copy.deepcopy(self)





# some general functions outside connectivity class
#-------------------------------------------------------------------------------------------------
def saveObj(obj, savePath, saveFileName):

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    with open(os.path.join(savePath, saveFileName+'.pkl'), 'wb') as fp:
        pickle.dump(obj, fp)
        print(saveFileName+'simuObj saved in'+savePath)












