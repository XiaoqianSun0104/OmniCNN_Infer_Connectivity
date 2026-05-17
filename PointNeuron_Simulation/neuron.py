'''
# neuron.py
# Author: Xiaoqian Sun, 06/04/2024
# a base neuron class, change neuronType to change corresponding parameters
'''


# Import Packages
import logging
import numpy as np

import signals

import warnings
warnings.filterwarnings('ignore')


class neuronParas(object):
    
    def __init__(self, T, dt, gE_bar=0.05, gI_bar=0.1, neuronType=0,
                 g_Leak=10., tau_m=15., tref=2., 
                 V_initMethod='constant',V_LeakReversal=-70., V_fireThreshold=-55., 
                 V_lowerbound=-100., V_reset=-70., V_init=-65.,
                 V_excReversal = 0., V_inhReversal =- 75., tau_excSyn=5., tau_inhSyn=10., 
                 tau_stdp=20., lr_stdp_preSyn=0.008, lr_stdp_postSyn=0.0088, 
                 lr_istdp=0.01, lr_istdp_window=0.12,
                 externalInput=None,
                 ifAssignSpkTrain=False, rate=15, assignSpkTrain=None, poissonSpk_seed=False, assignSynTrace=None,
                 ifadjust_spkTrain=False, ifARP=True, ifBursting=False, BP=0.8,
                 delayWindow=(2, 7),
                 g_AHP=0.15, V_AHP=-80., tau_Ca=100., alpha_Ca=0.2, #AHP related for bursting
                 **kwargs):
    
        self.T=T; self.dt=dt
        self.Nt = int(self.T/self.dt)
        self.neuronType = neuronType
        self.range_t = np.arange(0, self.T, self.dt)


        # initialize the paras{}
        self.paras = {
            'g_Leak': g_Leak,                     # nS (1μS=1000nS=10^6ps)
            'memResistance':(1/g_Leak)*1000,      # MΩ Rm = 1/gLeak * 1000
            'V_LeakReversal': V_LeakReversal,
            'V_fireThreshold': V_fireThreshold,   # -55mV~-50mV
            'V_lowerbound': V_lowerbound,
            'V_reset': V_reset,
            'V_initMethod':V_initMethod, 
            'V_init': V_init,
            'tau_m': tau_m,                       # the smaller, the faster respond; Tm = Cmem (membrane capacitance pF) * Rm
            'tref': tref,
            'Ntref': None,

            'g_AHP':g_AHP,                       # strength of AHP current; larger g_AHP, shorter bursts
            'V_AHP':V_AHP,                       # AHP channel equilibrium potential
            'tau_Ca':tau_Ca,                     # calcium decay; larger cau_Ca, slower bursting, smaller tau_ca, faster calcium decay, neuron recovers quickly
            'alpha_Ca':alpha_Ca,                 # how much calcium increases per spike; larger alpha_Ca, faster buildup of AHP, shorter burst duration

            'V_excReversal': V_excReversal,
            'V_inhReversal': V_inhReversal,
            'tau_excSyn':tau_excSyn,
            'tau_inhSyn':tau_inhSyn,
            'gE_bar':gE_bar, 
            'gI_bar':gI_bar, 
            'tau_stdp':tau_stdp, 
            'lr_stdp_preSyn':lr_stdp_preSyn,    # gamma
            'lr_stdp_postSyn':lr_stdp_postSyn,  # zeta
            'lr_istdp': lr_istdp,               # eta
            'lr_istdp_window':lr_istdp_window,  # alpha

            'externalInput': externalInput,

            'ifAssignSpkTrain':ifAssignSpkTrain, 
            'rate':rate,
            'ifadjust_spkTrain':ifadjust_spkTrain,
            'ifARP':ifARP,
            'ifBursting':ifBursting,
            'BP':BP,
            'delayWindow':delayWindow,
            'poissonSpk_seed':poissonSpk_seed,
            'assignSpkTrain':assignSpkTrain,  # pre-generates and assigned to this neuron
            'assignSynTrace':assignSynTrace,  # synaptic trace associated with the assignSpkTrain
            }
        
        # update
        parasKeys=self.paras.keys()
        for k,v in kwargs.items():
            if k in parasKeys:
                self.paras[k] = v
            else:
                logging.warning('No key in Object.paras named {0}'.format(k))
        
        self.update_attr()

    def update_attr(self):
        self.Nt = int(self.T/self.dt)
        self.range_t = np.arange(0, self.T, self.dt)


        self.paras['Ntref'] = int(round(self.paras['tref']/self.dt))

        if self.paras['V_initMethod'] == 'random':
            V_init = 10*np.random.random()+self.paras['V_reset']
            self.paras['V_init'] = V_init


        

        # generate presynaptic poisson spike train
        if self.paras['ifAssignSpkTrain'] and type(self.paras['assignSpkTrain'])==type(None):
            # poisson process
            assignSpkTrain = signals.PoissonSpkTrain(self.dt, self.range_t, self.paras['rate'], 1, myseed=self.paras['poissonSpk_seed'])  
            
            # iteratively adjust spikes to make the distribution a real exp
            if self.paras['ifadjust_spkTrain']:      
                assignSpkTrain = signals.adjust_expSpkTrain(assignSpkTrain, self.paras['rate'], self.T, self.dt, self.range_t)

            # add bursting activity
            if self.paras['ifBursting']: 
                assignSpkTrain = signals.add_burstActivity(assignSpkTrain, self.dt, self.paras['BP'], self.paras['delayWindow'])
            # modify by absolute refractory period
            if self.paras['ifARP']: 
                assignSpkTrain = signals.modify_absRefractory(assignSpkTrain, self.paras['Ntref'] )
            
            # store spkTrain to object
            self.paras['assignSpkTrain'] = assignSpkTrain

        if type(self.paras['assignSpkTrain']) == np.ndarray and type(self.paras['assignSynTrace'])==type(None):
            assignSynTrace = signals.generate_synapticTrace(self.paras['assignSpkTrain'], self.Nt, 
                                                            self.dt, self.paras['tau_stdp'])
            self.paras['assignSynTrace'] = assignSynTrace

        
        return(self)


    def get_keysValues(self):

        '''
        return all values stored in the object
            either direct attributes (obj.Nt)
            or dicts in the object (Obj.neuronPara)
        '''
        return (self.__dict__)
    
    def get_keys(self):
        '''
        return a list of attributes name, variables that can be accessed from the object
        '''
        return (self.__dict__.keys())
    
    def set(self, params_dict):
        """ 
        Set key-value pairs. E.g.,
            params_dict = {'V_excRevsal':16,  # in neuron.neuronPara{}
                           'Nt':6}    # neuron.Nt, direct attribute
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
                # if attrNotExist_Flag:
                #     logging.warning('No key in neuronParas Object named {0}'.format(k))
            else:
                setattr(self, k, v)

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
                logging.warning('No key in Object named {0}'.format(key))
        else:
            return(getattr(self, key))



class Neuron(neuronParas):

    ''' 
    variables that could be accessed:
    T, dt, Nt, range_t, neuronType, paras, dynamics
    '''

    def __init__(self, T, dt, gE_bar, gI_bar, neuronType,
                 g_Leak=10., tau_m=20., tref=2., 
                 V_initMethod='constant',V_LeakReversal=-70., V_fireThreshold=-50., 
                 V_lowerbound=-100., V_reset=-70., V_init=-65.,
                 V_excReversal = 0., V_inhReversal =- 75., tau_excSyn=3., tau_inhSyn=5., 
                 tau_stdp=20., lr_stdp_preSyn=0.008, lr_stdp_postSyn=0.0088, 
                 lr_istdp=0.01, lr_istdp_window=0.2,
                 externalInput=None,
                 ifAssignSpkTrain=False, rate=15, assignSpkTrain=None, poissonSpk_seed=False, assignSynTrace=None,
                 **kwargs
                 ):

        super().__init__(T, dt, gE_bar, gI_bar, neuronType, g_Leak, tau_m, tref, 
                         V_initMethod,V_LeakReversal, V_fireThreshold, V_lowerbound, V_reset, V_init,
                         V_excReversal, V_inhReversal, tau_excSyn, tau_inhSyn, tau_stdp, 
                         lr_stdp_preSyn, lr_stdp_postSyn, lr_istdp, lr_istdp_window,
                         externalInput,
                         ifAssignSpkTrain, rate, assignSpkTrain, poissonSpk_seed, assignSynTrace,
                         **kwargs)
        self.dynamics={
            'spkTrain':np.zeros(self.Nt), # generated during simulation
            'memPotential': np.ones(self.Nt),
            'synTrace': np.zeros(self.Nt), 
            
            'gE': np.zeros(self.Nt),
            'gI': np.zeros(self.Nt),
            
            'EPSP':np.zeros(self.Nt),
            'IPSP':np.zeros(self.Nt),
            'Xcurr':np.zeros(self.Nt),
            'LeakC':np.zeros(self.Nt),

            'spkTimes':[],
        }

        self.update_attr()


    def update_attr(self):
        super().update_attr()  
        self.dynamics={
            'spkTrain':np.zeros( self.Nt), # generated during simulation
            'memPotential': np.ones( self.Nt)*self.paras['V_init'],
            'synTrace': np.zeros(self.Nt), 
            
            'gE': np.zeros( self.Nt),
            'gI': np.zeros( self.Nt),
            
            'EPSP':np.zeros( self.Nt),
            'IPSP':np.zeros( self.Nt),
            'Xcurr':np.zeros(self.Nt),
            'LeakC':np.zeros( self.Nt),

            'spkTimes':[],
        }
        return self





