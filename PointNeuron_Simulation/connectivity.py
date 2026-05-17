'''
# connectivity.py
# Author: Xiaoqian Sun, 06/04/2024

# **** connectivity among neurons
#   recurrent connection within one subgroup (bi)
#   cross subgroups connection (bi)
#   external to one subgroup (one)
#   ...
# **** this should support differnet topology networks
'''


# Import Packages
import os 
import math
import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


import warnings
warnings.filterwarnings('ignore')



# connectivity matrix class
#-------------------------------------------------------------------------------------------------
class Connectivity(object):
    '''
    variables that could be accessed:
    N, Ne, Ni, paras{}, CM, CM_df, adjList, outgoingCs, incomingCs, CMW, CMW_df
        - CM, CM_df   : connectivity matrix/dataframe
        - adjList     : adjcent list from CM, call .print_adjList() to print it out
        - outgoingCs   : as a presynaptic neuron (A), connections from A to other neurons
        - incomingCs  : as a postsynaptic neuron (A), connections point to A
        - CMW, CMW_df : connectivity matrix with weights multiplied
    '''

    def __init__(self,
                 N, Ne, Ni, CM=None, cols=None,
                 p_exc2exc=0.2, p_exc2inh=0.2, p_inh2exc=0.2, p_inh2inh=0.2, 
                 we2e_max=1.5, we2i_max=1.5, wi2e_max=3, wi2i_max=3,
                 w_exc2exc=0.1, w_exc2inh=0.1, w_inh2exc=0.1, w_inh2inh=0.1,
                 **kwargs
                 ):
        

        # main assign
        self.N=N; self.Ne=Ne; self.Ni=Ni; self.CM=CM; self.cols=cols
        self.CMWParas = {'p_exc2exc':p_exc2exc, 'p_exc2inh':p_exc2inh, 
                         'p_inh2exc':p_inh2exc, 'p_inh2inh':p_inh2inh,
                         'w_exc2exc':w_exc2exc, 'w_exc2inh':w_exc2inh, 
                         'w_inh2exc':w_inh2exc, 'w_inh2inh':w_inh2inh,
                         'we2e_max':we2e_max,   'we2i_max':we2i_max, 
                         'wi2e_max':wi2e_max,   'wi2i_max':wi2i_max
                         }
        
        # update
        CMWParasKeys=self.CMWParas.keys()
        for k,v in kwargs.items():
            if k in CMWParasKeys:
                self.paras[k] = v
            elif k=='CM':
                self.CM=v
            else:
                logging.warning('No key in Object.CMWParas named {0}'.format(k))

        self.update_attr()


    def update_attr(self):
        if type(self.CM) == np.ndarray: # if connectivity matrix is given
            self.CM_df = M_to_df(self.CM, self.Ne, self.Ni, self.cols)
        else:
            self.CM, self.CM_df = self.generate_connectivityMatrix()
        
        # get connectivity attributes
        self.adjList = adjMatrix_2_adjList(self.CM)
        self.outgoingCs = adjMatrix_2_adjList(self.CM)
        self.incomingCs = adjMatrix_2_adjList(self.CM.T)

        # add weights to connectivity (0/1)
        self.CMW, self.CMW_df = self.addWeights()

    
    def generate_connectivityMatrix(self, ifVerbose=False):
    
        '''
        generate connectivity matrix
        each row is a single neuron and randomly select certain number of neurons to connect
        E.g.,
            - 1E, randomly select num=cee exc neurons to connect
            - 1E is the presynaptic neuron and it had cee outgoing connections to other exc neurons
        '''
        
        ce2e = int(self.CMWParas['p_exc2exc']*self.Ne)
        ce2i = int(self.CMWParas['p_exc2inh']*self.Ni)
        ci2e = int(self.CMWParas['p_inh2exc']*self.Ne)
        ci2i = int(self.CMWParas['p_inh2inh']*self.Ni)
        
        CM = np.zeros((self.N, self.N))
        for i in range(self.Ne):   
            CM[i, generate_randomInts(0, self.Ne-1, ce2e, i)] = 1
            CM[i, generate_randomInts(self.Ne, self.N-1, ce2i, i)] = 1

        for i in range(self.Ne, self.N):  
            CM[i, generate_randomInts(0, self.Ne-1, ci2e, i)] = 1
            CM[i, generate_randomInts(self.Ne, self.N-1, ci2i, i)] = 1
            
            
        # dataframe for better visualization
        cols = [str(i+1)+'E' for i in range(self.Ne)] + [str(i+1)+'I' for i in range(self.Ni)]
        CM_df = M_to_df(CM, self.Ne, self.Ni)
        
        if ifVerbose:
            print('There are', self.Ne, 'exc neurons and', self.Ni, 'inh neurons in the network')
            print('And we randomly choose pe2e⋅Ne(E2E)='+str(ce2e), 'pe2i⋅Ne(E2I)='+str(ce2i),
                'pi2e⋅Ni(I2E)='+str(ci2e), 'pi2i⋅Ni(I2I)'+str(ci2i), 'neurons to connect')
            print('That is: ce2e='+str(ce2e)+' | ce2i='+str(ce2i)+' | ci2e='+str(ci2e)+' | ci2i='+str(ci2i))
            
            
        return(CM, CM_df)
    
    def print_adjList(self):
        print_adjList(self.adjList)
    
    def addWeights(self):
        CMW, CMW_df = addWeight_2_connectivityMatrix(self.CM, self.CM_df, self.Ne, 
                                                    self.CMWParas['w_exc2exc'], self.CMWParas['w_exc2inh'], 
                                                    self.CMWParas['w_inh2exc'], self.CMWParas['w_inh2inh'])
        return(CMW, CMW_df)
    
    def get_keysValues(self):
        return (self.__dict__)
    
    def get_keys(self):
        return (self.__dict__.keys())
    
    def set(self, params_dict):
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
                # if attrNotExist_Flag:
                #     logging.warning('No key in Connectivity Object named {0}'.format(k))
            else:
                setattr(self, k, v)

        self.update_attr()

    def get(self, key):
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
                logging.warning('No key in Connectivity Object named {0}'.format(key))
        else:
            return(getattr(self, key))




# some general functions outside _connec class
#-------------------------------------------------------------------------------------------------
def generate_randomInts(start, stop, num, autapse, replace=False):
    candidates = [i for i in range(start, stop + 1) if i != autapse]

    if len(candidates) == 0:
        raise ValueError("No valid candidates available after excluding autapse.")

    if not replace and num > len(candidates):
        raise ValueError(f"Cannot sample {num} unique integers from only {len(candidates)} valid candidates.")

    if replace:
        rand_ints = random.choices(candidates, k=num)
    else:
        rand_ints = random.sample(candidates, k=num)

    return sorted(rand_ints)

def generate_randomConnection(preSyn_N, postSyn_N, connecP=0.5, weight=None, ifAutapse=True):
    
    local_connectivity = np.zeros((preSyn_N, postSyn_N))
    numConnection = int(connecP * postSyn_N)
    
    for i in range(preSyn_N):
        if ifAutapse:
            local_connectivity[i, generate_randomInts(0, postSyn_N-1, numConnection, i)] = 1
        else:
            local_connectivity[i, generate_randomInts(0, postSyn_N-1, numConnection, None)] = 1
            
    return local_connectivity
    
def generate_normDistConnection(preSyn_N, postSyn_N, mean, std, ifShuffle=True):
    
    local_connectivity = np.zeros((preSyn_N, postSyn_N))
    
    for i in range(preSyn_N):
        weights = np.random.normal(mean, std, postSyn_N)
        if ifShuffle:
            np.random.shuffle(weights)
        
        local_connectivity[i] = weights
        
    local_connectivity[local_connectivity<0] = 0.05
    
    return (local_connectivity)
    
def M_to_df(connections, Ne, Ni, cols=None, dataType=float):
    '''
    form the matrix to a dataframe for better visualization
    '''
    if type(cols)==type(None):
        cols = [str(i+1)+'E' for i in range(Ne)] + [str(i+1)+'I' for i in range(Ni)]
    CM_df = pd.DataFrame(connections, columns=cols, index=cols).astype(dataType)
    return (CM_df)

def adjMatrix_2_adjList(connections):
        # make sure adjMatrix is an array
        import numpy as np
        if type(connections) != np.ndarray:
            connections = np.array(connections)
        
        adjList = defaultdict(list)
        for i in range(len(connections)):
            
            adjList[i] = list(np.where(connections[i] != 0)[0])
            
            
        return (adjList)

def print_adjList(adjList):
    for i in adjList:
        print(i, end ="")
        
        for j in adjList[i]:
            print("-> {}".format(j), end =" ")
        
        print()
    
def get_connection_ExcInh(connections, neuronIndex, Ne):
    conn = np.array(connections[neuronIndex])
    
    idx_exc = list(conn[conn<Ne])
    idx_inh = list(conn[conn>=Ne])
    
    return (idx_exc, idx_inh)

def connecHalf_randomC(local_connec, connectP=None, connectN=None):
    pre_num, post_num = local_connec.shape
    
    if connectP:
        pre2post_c_num = int(post_num*connectP)
    
    if connectN:
        pre2post_c_num = int(connectN)

    if type(connectP)==type(None) and type(connectN)==type(None):
        raise ValueError('Must assign connection probability or number of connections')
        
    for i in range(pre_num):   
        local_connec[i, generate_randomInts(0, post_num-1, pre2post_c_num, None)] = 1
        
    
    return (local_connec)
    
def addWeight_2_connectivityMatrix(CM, CM_df, Ne, we2e, we2i, wi2e, wi2i):
    
    CMW = CM.copy()
    CMW[0:Ne, 0:Ne] = CMW[0:Ne, 0:Ne]*we2e
    CMW[0:Ne, Ne:] = CMW[0:Ne, Ne:]*we2i
    CMW[Ne:, 0:Ne] = CMW[Ne:, 0:Ne]*wi2e
    CMW[Ne:, Ne:] = CMW[Ne:, Ne:]*wi2i
    
    # form a dataframe
    CMW_df = pd.DataFrame(CMW, columns=CM_df.columns, index=CM_df.index)
    
    return(CMW, CMW_df)

def CMW_upperBound(CMW, Ne, we2e_max, we2i_max, wi2e_max, wi2i_max):
    '''
    make sure the weights don't go beyond the upper bound
    this case can be called as 'saturation'
    '''
    
    CMW[0:Ne, 0:Ne][np.where(CMW[0:Ne, 0:Ne]>we2e_max)] = we2e_max
    CMW[0:Ne, Ne:][np.where(CMW[0:Ne, Ne:]>we2i_max)] = we2i_max

    CMW[Ne:, 0:Ne][np.where(CMW[Ne:, 0:Ne]>wi2e_max)] = wi2e_max
    CMW[Ne:, Ne:][np.where(CMW[Ne:, Ne:]>wi2i_max)] = wi2i_max
    
    return(CMW)

def get_changingWeights(CMW_rlist, incomingCs, neuronIdx, Ne):
    '''
    here, neuronIdx is the postsynaptic neuron, get its presynaptic neurons' weights from CMW_rlist
    '''
    
    pre_exc, pre_inh = get_connection_ExcInh(incomingCs, neuronIdx, Ne)
    
    
    WE = []
    for exc in pre_exc:
        we = [cmw[exc,neuronIdx] for cmw in CMW_rlist]
        WE.append(we)
    
    WI = []
    for inh in pre_inh:
        wi = [cmw[inh,neuronIdx] for cmw in CMW_rlist]
        WI.append(wi)

    return(WE, WI)

def clip_externalNeuron(CMW, Xe, Xi, Se, Si):
    '''
    in a network, there will be some neuron got assgined spike trains and provide drive to the networks
    these neurons don't receive input (synapses) from other neurons, so we mgith want to clip them
    note that these neurons need to be at the beginning of each section
    e.g., Xe=10 (10 exc providing exc drive) and idx: 0 - 9
          Xi=10 (10 inh providing inhibition) and idx: Xe+Se+0 - Xe+Se+9
    And since connectivity matri is a square matrix, we need to remove bothe rows and cols (same index)

    Arguments:
        - Xe: number of exc external neuron
        - Xi: number of inh external neuron
        - Se: number of exc simulating neuorn
        - Si: numer of inh simulating neuron
    '''

    columns_to_remove = list(range(0, Xe)) + list(range(Xe+Se, Xe+Se+Xi))
    CM_filtered = np.delete(CMW, columns_to_remove, axis=1)
    CM_filtered_filtered = np.delete(CM_filtered, columns_to_remove, axis=0)

    return CM_filtered_filtered







