#!/usr/bin/env python
# peelParams.py
#
# Author: Xiaoqian Sun, 07/2023 
#
# Parameters used in 2nd part, including peeling and reconstruction. 
# The parameters are for each neuron and some are inherited from simuParams.simulation
# Ref: https://github.com/flatironinstitute/CaImAn/blob/main/caiman/source_extraction/cnmf/params.py
'''
some consistency in simuParameters and peelParameters
ca_p['amp1'] = A1[m]     - use A1
ca_p['amp2'] = A2[m]     - use A2  
ca_p['onsettau'] = tauOn[m]    - use tauOn
exp_p['noiseSD'] = sdnoise[m]  - use noiseSD

'''


# Import Packages
#========================================================================================
import logging
import numpy as np
from pprint import pformat


class peelParams(object):
    """
    Class for setting the simulation process parameters. 
    All parameters for DFF simulation, calcium dynamics, experiment, peeling algorithm can be set here and then 
    used in various in the various processing pipeline steps.
    Any parameter that is not set get a default value specified by the dictionary default options
    """
    
    def __init__(self, dff,
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
                 
                 params_dict={} ):
        
        dff_length = len(dff)
        # update some values
        if ca_genmode == 'satDFF':
            ca_amp1 = ca_amp/(1+ca_kappas+kappab)
            ca_tau1 = (1+ca_kappas+kappab)/ca_gamma
        kappab = kd*conc/(ca_rest+kd)**2
        # smtthigh = schmitt[0]*noiseSD
        # smttlow = schmitt[1]*noiseSD
        # smttmindur = schmitt[2]


        # Parameters of elementary (1 AP) calcium transient (calcium dynamics)  
        self.ca_p = {
            'ca_genmode': ca_genmode,          # flag for spike generation mode: 'linDFF' - simple linear DFF, or 'satDFF' - saturating indicator
            'ca_amp': ca_amp,                  # Ca transient - ingle AP Ca transient amplitude (nM）
            'ca_gamma': ca_gamma,              # Ca transient - extrusion rate (1/s)
            'ca_amp1': ca_amp1,                # Ca transient - free Ca amplitude 1  (nM)
            'ca_tau1': ca_tau1,                # Ca transient - free Ca tau (s) (decay time constant)
            'ca_kappas': ca_kappas,            # Ca transient - endogenous Ca-binding ratio 
            'ca_rest': ca_rest,                # presumed resting calcium concentration (nM)
            'ca_current': ca_current,          # current calcium concentration (nM)   
            'ca_onsettau': ca_onsettau,        # Ca transient - onset tau (s)
            'onsetposition': onsetposition,

            'A1': A1,                          # single AP DFF in % 
            'tau1': tau1,                      # indicator decay time in second
            'A2': A2,                          # second amplitude for double-exp decay, set to 0 
            'tau2': tau2,                      # decay time for second exponential
            'tauOn': tauOn,                    # tauOn, onset time in s, which is onsettau
            'offset': offset,
            'scale': scale,                    # scale factor to scale entire trace (s)
            'integral': integral,              # integral below curve (#s)
            'negintegral': negintegral, }
        
        
        self.exp_p = {
            'dffmax': dffmax,                  # saturating dff max (in percent)
            'numpnts': dff_length,             # num points
            'frameRate': frameRate,            # acquisition rate (Hz) (frame rate)
            'noiseSD': noiseSD,                # noise stdev of DF/F trace (in percent), should be specified by the user
            'indicator': indicator,            # calcium indicator
            
            'kd': kd,                          # dye dissociation constant (nM)
            'conc': conc,                      # dye total concentration (nM)
            'kappab': kappab, }
        
        
        self.peel_p = {
            'spk_recmode': spk_recmode,         # flag, for spike reconstruction mode: 'linDFF', or 'saturatDFF'       
            
            'smtthigh': smtthigh,               # Schmitt trigger - high threshold (multiple of exp_p.noiseSD),   
            'smttlow': smttlow,                 # Schmitt trigger - low threshold (multiple of exp_p.noiseSD),   
            'smttbox': smttbox,                 # Schmitt trigger - smoothing box size (in points)
            'smttmindur': smttmindur,           # Schmitt trigger - minimum duration (s)
            'smttnumevts': smttnumevts,         # Schmitt trigger - number of found events
            'smttmindurFrames': smttmindurFrames,              # how long an event should last            
            'smttlowMinEvents': smttlowMinEvents,              # the value that an event fall below meaning end of the event 
            
            'padding': padding,                 # number of points for padding before and after
            'slidwinsiz': slidwinsiz,           # sliding window size - event detection (s)               
            'maxbaseslope': maxbaseslope,       # maximum baseslope #/s  
            
            'evtfound': evtfound,               # flag - 1: crossing found
            'nextevt': nextevt,                 # next crossing found (s)
            'nextevtframe': nextevtframe,       # next crossing found (frame number)
            'evtaccepted': evtaccepted,         # if accept the event

            'intcheckwin': intcheckwin,         # window to the right - for integral comparison (s) 
            'intacclevel': intacclevel,         # event integral acceptance level (0.5 means 50#)
            'fitonset': fitonset,               # flag - T: do onset fit, only useful if 1/frameRate <= rise of CacliumTransient
            'fitwinleft': fitwinleft,           # left window for onset fit (s) 
            'fitwinright': fitwinright,         # right window for onset fit (s)
            'negintwin': negintwin,             # window to the right - for negativeintegral check(s)
            'negintacc': negintacc,             # negative acceptance level (0.5 means 50#)
            'stepback': stepback,               # stepsize backwards for next iteration (s)       
            'fitupdatetime': fitupdatetime,     # how often the linear fit is updated (s)      
            
            'optimizeSpikeTimes': optimizeSpikeTimes,            # if optimize reconstructed spike times to improve timing         
            'doPlot': doPlot, }


        self.data = {

            'dff': dff,
            'peel': dff,

            'intdff': np.arange(dff_length),              # integral curve
            'temp': np.arange(dff_length),                # temporary wave

            'freeca': np.zeros(dff_length),               # free calcium transient, from which dff will need to be calculated 
            'singleTransient': np.ones(dff_length)*0.1,      # fluorescence transient for current AP, will take ca2fluor mode into account
            'freecamodel': np.zeros(dff_length), 
            'model': np.zeros(dff_length), 
            'modelConvolve': np.zeros(dff_length), 
            'spiketrain': np.zeros(dff_length), 
            'slide': np.zeros(dff_length), 

            'tim': np.arange(dff_length)/self.exp_p['frameRate'], 

            'spikes': np.zeros(int(1e6)),                     # array for found spikes times
            'numspikes': 0,                              # number of spikes found

            'dff_pad': np.array([]),
            'peel_pad': np.array([]), 
            'slide_pad': np.array([]), 
            'temp_pad': np.array([]), 
            'tim_pad': np.array([]), }

        self.change_params(params_dict)


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
            raise KeyError('No group in peelParams named {0}'.format(group))

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
        return {'ca_p': self.ca_p, 'exp_p': self.exp_p, 'peel_p': self.peel_p, 'data': self.data}
    
    def __repr__(self):

        formatted_outputs = [
            '{}:\n\n{}'.format(group_name, pformat(group_dict))
            for group_name, group_dict in self.to_dict().items()
        ]

        return 'peelParams:\n\n' + '\n\n'.join(formatted_outputs)

    def change_params(self, params_dict, verbose=False):
        """ Method for updating the params object by providing a single dictionary.
        For each key in the provided dictionary the method will search in all
        subdictionaries and will update the value if it finds a match.

        Args:
            params_dict: dictionary with parameters to be changed and new values
            verbose: bool (False). Print message for all keys
        """
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

