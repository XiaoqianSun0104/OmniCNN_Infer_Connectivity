#!/usr/bin/env python
# modelCa2Params.py
#
# Author: Xiaoqian Sun, 07/2023 
#
# Parameters used in 1st part. 
# Ref: https://github.com/flatironinstitute/CaImAn/blob/main/caiman/source_extraction/cnmf/params.py
# The self parameter is a reference to the current instance of the class, and is used to access variables that belong to the class.


# Import Packages
#========================================================================================
import logging
import numpy as np
from pprint import pformat


class ModelCa2Params(object):
    """f
    Aim:
        Class for setting the simulation process parameters. 
        All parameters for DFF simulation, calcium dynamics, experiment, peeling algorithm can be set here and then 
        used in various in the various processing pipeline steps.
        Any parameter that is not set get a default value specified by the dictionary default options

    Note:
        - if a model, [A1, tau1, A2, tau2, tauOn, spikeRate, snr] need to assign same amount of values
    """
    
    def __init__(self, dur, cellNum,
                 spk_recmode = 'linDFF', offset=0, doPlot=True, doVectorized=False, 
                 ifAssignSpk = False, spkTimes=[],spkTrain=[],
                 spikeRate = np.array([8]), ca_genmode = 'linDFF', ca_amp = np.array([7600]), ca_amp1 = np.array([0]),
                 ca_tau1 = np.array([0]), ca_rest = np.array([50]), ca_gamma = np.array([400]), ca_kappas = np.array([100]),
                 ca_onsettau = np.array([0.02]), kd = 250, conc = 50000, kappab =0,
                 A1 = np.array([1.5]), A2 = np.array([0]), tau1 = np.array([0.1]),
                 tau2 = np.array([0.03]), tauOn = np.array([0.01]), A1sigma = np.array([]), tau1sigma = np.array([]),
                 snr = np.array([5]), dffmax = np.array([93]), frameRate =30, samplingRate = 1000,
                 recycleSpikeTimes = False, **kwargs):
        
        # update some values
        if ca_genmode == 'satDFF':
            ca_amp1 = ca_amp/(1+ca_kappas+kappab)
            ca_tau1 = (1+ca_kappas+kappab)/ca_gamma
        kappab = kd*conc/(ca_rest+kd)**2 # an array, kappab should be a float
        kappab = kappab[0]

        self.ifAssignSpk = ifAssignSpk
        self.simulation = {
            'dur': dur,
            'offset': offset,
            'doPlot': doPlot,
            'doVectorized': doVectorized,
            'cellNum': cellNum,

            'spikeRate': spikeRate,

            'ca_genmode': ca_genmode,
            'ca_amp': ca_amp,
            'ca_amp1': ca_amp1,
            'ca_tau1': ca_tau1,
            'ca_rest': ca_rest,
            'ca_gamma': ca_gamma,
            'ca_kappas': ca_kappas,
            'ca_onsettau': ca_onsettau,

            'kd': kd,
            'conc': conc,
            'kappab': kappab,

            'A1': A1,
            'A2': A2,
            'tau1': tau1,
            'tau2': tau2,
            'tauOn': tauOn,
            'A1sigma': A1sigma,
            'tau1sigma': tau1sigma,

            'snr': snr,
            'dffmax': dffmax,
            'peakA':np.zeros(cellNum),
            'noiseSD':np.zeros(cellNum),
            'spkTimes':spkTimes,
            'spkTrain':spkTrain,
           

            'frameRate':frameRate,
            'samplingRate':samplingRate,
            'recycleSpikeTimes':recycleSpikeTimes,

            'spk_recmode':spk_recmode,
            }

        # update
        parasKeys=self.simulation.keys()
        vectorizedKeys = ['A1',  'A2', 'tau1', 'tau2', 'tauOn', 'ca_amp', 'ca_gamma', 'ca_onsettau', 'ca_rest', 'ca_kappas', 'ca_rest', 'dffmax', 'noiseSD', 'snr']
        if doVectorized:
            for k in vectorizedKeys:
                if k in parasKeys:
                    self.simulation[k] = np.repeat(self.simulation[k], cellNum)

        for k,v in kwargs.items():
            if k in parasKeys:
                if doVectorized and k in vectorizedKeys:
                    self.simulation[k] = np.repeat(v, cellNum)
                else:
                    self.simulation[k] = v
            else:
                logging.warning('No key in Object.paras named {0}'.format(k))
        self.vectorize('simulation')

        # print('ifAssignSpk =',ifAssignSpk)
        if self.ifAssignSpk:
            self.simulation['spkTimes'] = spkTimes
            self.simulation['spkTrain'] = spkTrain


    def set(self, group, val_dict, set_if_not_exists=False, verbose=False):
        """ 
        Add key-value pairs to a group. 
        Existing key-value pairs will be overwritten if specified in val_dict, but not deleted.

        Args:
            group: The name of the group.
            val_dict: A dictionary with key-value pairs to be set for the group.
        """

        if not hasattr(self, group):
            raise KeyError('No group in peelParams named {0}'.format(group))

        d = getattr(self, group)
        for k, v in val_dict.items():
            if k not in d and not set_if_not_exists:
                if verbose:
                    logging.warning("NOT setting value of key {0} in group {1}, because no prior key existed...".format(k, group))
            else:
                if np.any(d[k] != v):
                    logging.info("Changing key {0} in group {1} from {2} to {3}".format(k, group, d[k], v))
                d[k] = v

    
    def get(self, group, key):
        """ 
        Get a value for a given group and key. Raises an exception if no such group/key combination exists.

        Args:
            group: The name of the group.
            key: The key for the property in the group of interest.

        Returns: The value for the group/key combination.
        """

        if not hasattr(self, group):
            raise KeyError('No group in ModelCa2Params named {0}'.format(group))

        d = getattr(self, group)
        if key not in d:
            raise KeyError('No key {0} in group {1}'.format(key, group))

        return d[key]

    
    def get_group(self, group): 
        return getattr(self, group)

    
    def to_dict(self):
        """
        Returns the params class as a dictionary with subdictionaries for eachcatergory
        """
        return {'simulation': self.simulation}

    
    def __repr__(self):

        formatted_outputs = [
            '{}:\n\n{}'.format(group_name, pformat(group_dict))
            for group_name, group_dict in self.to_dict().items()
        ]

        return 'ModelCa2Params:\n\n' + '\n\n'.join(formatted_outputs)


    def change_params(self, params_dict, verbose=False):
        """ Method for updating the params object by providing a single dictionary.
        For each key in the provided dictionary the method will search in all
        subdictionaries and will update the value if it finds a match.

        Args:
            params_dict: dictionary with parameters to be changed and new values
            verbose: bool (False). Print message for all keys
        """
        # if a model or single neuron
        argVector = ['A1', 'tau1', 'A2', 'tau2', 'tauOn', 'spikeRate', 'snr'] # make sure these are of same length
        if len( set( [len(params_dict[l]) for l in argVector ] ) ) > 1:
            raise Exception('Argument lengths are inconsistent')
        
        for gr in list(self.__dict__.keys()):
            self.set(gr, params_dict, verbose=verbose)
        for k, v in params_dict.items():
            flag = True
            for gr in list(self.__dict__.keys()):
                d = getattr(self, gr)
                if k in d:
                    flag = False
            if flag:
                logging.warning('No parameter {0} found!'.format(k))
        
        return self


    def vectorize(self, group):
        d = getattr(self, group)
        for k in ['ca_amp', 'ca_amp1', 'ca_tau1', 'ca_rest', 'ca_gamma', 'ca_kappas', 'ca_onsettau', 'dffmax', 'noiseSD']:
            if len(d[k]) != self.simulation['cellNum']:
                d[k] = np.repeat(d[k], self.simulation['cellNum'])
        
        # each neuron has a spkTimes list
        if not self.ifAssignSpk:
            d['spkTimes'] = [[] for i in range(self.simulation['cellNum'])]
            d['spkTrain'] = [[] for i in range(self.simulation['cellNum'])]

        return self
    

    def get_valuesByIndex(self, group, neuronIndex): 
   
        '''
        This is to get certain neuron's values which are used to construct peelPara obj
        '''
        key_list = ['ca_genmode', 'A1', 'tau1', 'A2', 'tau2', 'tauOn',
                'ca_amp', 'ca_amp1','ca_tau1','ca_rest','ca_gamma', 'ca_kappas', 'ca_onsettau', 
                'dffmax', 'noiseSD', 'kd', 'kappab', 'spk_recmode']
        
        paraI = {}
        d = getattr(self, group)
        for k in key_list:
            if isinstance(d[k], (str,np.int64, np.int32, np.float64, float, np.float32, int)):
                paraI[k] = d[k]
            elif isinstance(d[k], np.ndarray):
                paraI[k] = d[k][neuronIndex]
        
        return(paraI)








