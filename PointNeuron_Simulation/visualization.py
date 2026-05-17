'''
# visualization.py
# Author: Xiaoqian Sun, 06/04/2024
# functions to generate plots
'''


# Import Packages
#========================================================================================
import os 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from utils import *
import connectivity

import warnings
warnings.filterwarnings('ignore')



# functions
#========================================================================================
def plot_synTrace_spkTrain(spkTrain, synTrace, range_t, c='g', ifSave=False, savePath=None, filename=None):
    
    t_sp = range_t[spkTrain > 0.5]   # spike times
    
    plt.figure(figsize=(20, 2))
    plt.plot(t_sp, max(synTrace)*1.5*np.ones(len(t_sp)), '|', color=c, ms=20, markeredgewidth=2)
    plt.plot(range_t, synTrace, color=c, lw=2)
    # plt.xlim(-2, max(range_t)+2)
    plt.title('neuron synaptic trace associated with spkTrain')

    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_spkTrain(spkTrain_array, firing_rates, num2Plot, range_t, ifSave=False, savePath=None, filename=None):
    
    '''
    show spk trains generated based on different firing rates
    
    '''
    plt.figure(figsize=(20, 4))
    for i in range(num2Plot):
        FR = round(firing_rates[i], 3)
        t_sp = range_t[spkTrain_array[i, :] > 0.5]   # spike times
        plt.plot(t_sp, i*np.ones(len(t_sp)), '|',ms=20, markeredgewidth=3, label='FR='+str(FR))
    
    plt.legend()
    
    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_CM(CM_df, nums, labels, ifSave=False, savePath=None, filename=None):
    
    '''
    plot connectivity matrix to visulize connections between neurons
    
    '''

    plt.figure(figsize=(6, 4))
    sns.heatmap(CM_df, cmap='Blues', cbar=True, cbar_kws = dict(use_gridspec=False,location="left"))

    # draw dividing lines
    for i in range(1, len(nums)):
        plt.axvline(x=np.sum(nums[:i]), color='b')
        plt.axhline(y=np.sum(nums[:i]), color='b')
    
    # add text
    lIdx = 0
    num_accu_sum = np.array(accumulated_sum(nums))
    for y in num_accu_sum:
        for i in range(len(nums)):
            x = np.sum(nums[:i])
            plt.text(int(x), int(y), labels[lIdx], c='brown', fontsize=12, fontweight='semibold')
            lIdx+=1

    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()



def plot_1Trace(neuronObj, key, ifSave=False, savePath=None, filename=None):
    '''
    plot one neuron membrane potential trace
    '''
    plt.figure(figsize=(20, 3))
    plt.plot(neuronObj.get(key), lw=2)
    
    # save/show
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_1NTraces(neuronObj, keys, ifSave=False, savePath=None, filename=None):
    '''
    plot one neuron membrane potential trace
    '''

    r=len(keys)
    fig, ax = plt.subplots(r, 1, figsize=(20, 2*r))
    for i in range(r):
        key = keys[i]
        ax[i].plot(neuronObj.get(key), lw=2, label=key)
        ax[i].legend(loc=1)
    
    # save/show
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_NsTraces(neuronObjs, key, color='b', N=None, ifSave=False, savePath=None, filename=None):
    '''
    plot one neuron membrane potential trace
    '''

    if N:
        r=N; c=1
    else:

        r=len(neuronObjs); c=1
    fig, ax = plt.subplots(r, 1, figsize=(20, 2*r))
    for i in range(r):
        neuronObj = neuronObjs[i]
        ax[i].plot(neuronObj.get(key), lw=2, c=color, label='neuron '+str(i)+' '+key)
        ax[i].legend(loc=1)
    
    # save/show
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_spkTrain_objs(obj_list, firing_rates, num2Plot, range_t, ifSave=False, savePath=None, filename=None):
    '''
    show spk trains generated based on different firing rates
    Note that the input is not spkTrain 2D array, but a object list
    '''
    
    plt.figure(figsize=(20, 3))
    
    for i in range(num2Plot):
        FR = round(firing_rates[i], 3)

        spkTrain = obj_list[i].get('assignSpkTrain')
        t_sp = range_t[spkTrain > 0.5]   # spike times
        plt.plot(t_sp, i*np.ones(len(t_sp)), '|',ms=20, markeredgewidth=3, label='FR='+str(FR))
    
    plt.legend()
    
    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_spkTrain_synTrace_objs(obj_list, firing_rates, num2Plot, ifSave=False, savePath=None, filename=None):
    '''
    show spk trains generated based on different firing rates
    Note that the input is not spkTrain 2D array, but a object list
    '''
    
    if not isinstance(obj_list, list):
        raise ValueError('The input obj_list/firing_rates must be list')
    
    
    if num2Plot < 2:

        plt.figure(figsize=(20, 2))

        synTrace = obj_list[0].get('assignSynTrace')
        synTraceMax = synTrace.max()+0.4
        plt.plot(synTrace, lw=2, alpha=0.8, c='b')
        
        spkTrain = obj_list[0].get('assignSpkTrain')
        t_sp = np.where(spkTrain > 0.5)   # spike times
        plt.plot(t_sp, synTraceMax*np.ones(len(t_sp)), '|', ms=10, markeredgewidth=3, c='b')
    
        plt.title('neuron 0 | FR='+str(round(firing_rates[0], 3)), fontsize=14)
    else:

        fig, ax = plt.subplots(num2Plot, 1, figsize=(20, 2*num2Plot))
        for i in range(num2Plot):
            FR = round(firing_rates[i], 3)

            synTrace = obj_list[i].get('assignSynTrace')
            synTraceMax = synTrace.max()+0.4
            ax[i].plot(synTrace, lw=2, alpha=0.8, c='b')
            
            spkTrain = obj_list[i].get('assignSpkTrain')
            t_sp = np.where(spkTrain > 0.5)   # spike times
            ax[i].plot(t_sp, synTraceMax*np.ones(len(t_sp)), '|', ms=10, markeredgewidth=3, c='b')
        
            ax[i].set_title('neuron '+str(i)+' | FR='+str(FR), fontsize=14)
    
    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_neuronSynSources(Obj_list, targetN_idx, targetN_name, Ne, incomingConnections, R=6, C=1, ifVerbose=True, ifSave=False, savePath=None, filename=None):
    
    '''
    a subplots which contains target neuron's: membrane potential | EPSP | IPSP | Xcurr | gE | gI
        
    '''
    
    targetObj = Obj_list[targetN_idx]
    pre_exc, pre_inh = connectivity.get_connection_ExcInh(incomingConnections, targetN_idx, Ne)
    if ifVerbose: 
        print(targetN_name+str(targetN_idx)+' receives Exc from', pre_exc, 'and Inh from ', pre_inh); print()

        
    # plot
    fig, ax = plt.subplots(R, C, figsize=(20, 2*R))

    ax[0].plot(targetObj.get('memPotential'), lw=2); ax[0].set_title('memPotential', fontweight='bold')

    ax[1].plot(targetObj.get('EPSP')); ax[1].set_title('EPSP (Exc Input)', fontweight='bold')

    ax[2].plot(targetObj.get('IPSP')); ax[2].set_title('IPSP (Inh Input)', fontweight='bold')
    
    ax[3].plot(targetObj.get('Xcurr')); ax[3].set_title('Xcurr (External Input)', fontweight='bold')

    ax[4].plot(targetObj.get('gE')); ax[4].set_title('conductance gExc', fontweight='bold')
    
    ax[5].plot(targetObj.get('gI')); ax[5].set_title('conductance gInh', fontweight='bold')
    
    plt.tight_layout()
    # save
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()



def subplots_heatmaps(df_list, label_list,
                      yticklabels_list, xticklabels_list, figR, figC, figSize, annot=False, 
                      df_Rstart=10, df_Rend=20, df_Cstart=10, df_Cend=20, 
                      vMin=-1, vMax=1, cmap='jet', ifSave=True, savePath=None, filename=None):


    '''
    plot heamtpas in subplots format
    Args:
        - df_list, at least 2 dfs
        - label_list, corresponding df label
        - yticklabels_list/xticklabels_list: if (T/F) to show col/row name for each subplot; if use int, will plot every nth label
        - figR/figC: subplots arrangements
        - figSize
        - df_numRows/df_numCols: portion of df to show in heatmap (e.g., heatmap of whole df OR heatmap of first 10 rows and cells)
        - annot: if mark values on heatmap 
    '''

    # precheck
    if figR*figC != len(label_list):
        raise ValueError ('number of dfs does not match number of subplots!')
    

    if figR==1 or figC==1:
        fig, axs = plt.subplots(figR, figC, figsize=figSize)
        for i in range(len(label_list)):
            df = df_list[i].iloc[df_Rstart:df_Rend, df_Cstart:df_Cend]
            sns.heatmap(df, cmap=cmap, square=False, annot=annot, vmin=vMin, vmax=vMax, xticklabels=xticklabels_list[i], 
                        yticklabels=yticklabels_list[i], cbar=False, ax=axs[i])
            axs[i].set_title(label_list[i], fontweight='bold')
            axs[i].set_yticklabels(axs[i].get_yticklabels(), fontweight='bold')
            axs[i].set_xticklabels(axs[i].get_xticklabels(), fontweight='bold')
    else:
        dfIdex = 0
        fig, axs = plt.subplots(figR, figC, figsize=figSize)
        for r in range(figR):
            for c in range(figC):
                df = df_list[dfIdex].iloc[df_Rstart:df_Rend, df_Cstart:df_Cend]
                sns.heatmap(df, cmap=cmap, square=False, annot=annot, vmin=vMin, vmax=vMax, xticklabels=xticklabels_list[dfIdex], 
                        yticklabels=yticklabels_list[dfIdex], cbar=False, ax=axs[r, c])
                axs[r,c].set_title(label_list[dfIdex], fontweight='bold')
                axs[r,c].set_yticklabels(axs[r,c].get_yticklabels(), fontweight='bold')
                axs[r,c].set_xticklabels(axs[r,c].get_xticklabels(), fontweight='bold')
                dfIdex+=1

    plt.tight_layout()
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def subplots_histos(data_list, label_list, color_list, figR, figC, figSize, density=False, bins=30, ifSave=True, savePath=None, filename=None):
    '''
    plot several histograms in a subplot
    '''

    if figR==1 or figC==1:
        fig, ax = plt.subplots(figR, figC, figsize=figSize)
        for i in range(int(figR*figC)):
            data = data_list[i]
            data_mean = round(np.mean(data), 2); data_median = round(np.median(data), 2)
            ax[i].hist(data, bins=bins, facecolor=color_list[i], alpha=0.6, density=density)
            ax[i].set_title(f'{label_list[i]}, mean={data_mean}, median={data_median}', fontsize=12, fontweight='bold')
        plt.tight_layout()
    
    else:
        fig, ax = plt.subplots(figR, figC, figsize=figSize)
        idx = 0
        for r in range(figR):
            for c in range(figC):
                data = data_list[idx]
                data_mean = round(np.mean(data), 2); data_median = round(np.median(data), 2)
                ax[r, c].hist(data, bins=bins, facecolor=color_list[idx], alpha=0.6, density=density)
                ax[r, c].set_title(f'{label_list[idx]}, mean={data_mean}, median={data_median}', fontsize=12, fontweight='bold')
                idx+=1
        plt.tight_layout()
        
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

            

def plot_preSpkTrain_postMemPotential(nIndex, preSpkTrain, postMemPotential, ifSave=True, savePath=None, filename=None):
    '''
    plot presynaptic spkTrain on top and corresponding postsynaptic membrane potential
    '''

    if type(nIndex)==int:
        plt.figure(figsize=(20, 2))
    
        t_sp = np.where(preSpkTrain.iloc[:,nIndex] > 0.5)
        plt.plot(t_sp, -40*np.ones(len(t_sp)), '|', ms=10, markeredgewidth=3, c='brown')
        plt.plot(postMemPotential.iloc[:,nIndex], c='b', lw=1); plt.title('Pair'+str(nIndex), fontsize=14, fontweight='bold')
        
        plt.tight_layout()
    
    elif type(nIndex)==list:
        fig, ax = plt.subplots(len(nIndex), 1, figsize=(20, 2*len(nIndex)))
        
        for i in range(len(nIndex)):
            nIdx = nIndex[i]
            t_sp = np.where(preSpkTrain.iloc[:,nIdx] > 0.5)
            ax[i].plot(t_sp, -40*np.ones(len(t_sp)), '|', ms=10, markeredgewidth=3, c='brown')
            ax[i].plot(postMemPotential.iloc[:,nIdx], c='b', lw=1); ax[i].set_title('Pair'+str(nIdx), fontsize=14, fontweight='bold')
        plt.tight_layout()

    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()



def plot_ccg(ccg, timeBins, figsize=(6, 4), barColor='peru', alpha=0.8, title=None, ifSave=False, savePath=None, filename=None):

    '''
    plot ccg as a bar plot - a lot like figures in papers
    '''
    
    plt.figure(figsize=figsize)
    plt.bar(timeBins, ccg, width=1, color=barColor, alpha=alpha, edgecolor='none')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tick_params(direction='out')
    
    
    plt.xlabel('Time lag [s]'); plt.ylabel('Count'); plt.title(title)

    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename), dpi=300, bbox_inches="tight", transparent=True)
        plt.close()
    else:
        plt.show()
    
def plot_dcCCH_process(preName, postName, ach1, ach1_normed, freqs_ach1, fft_ach1, ach2, ach2_normed, 
                       freqs_ach2, fft_ach2, cch, freqs_cch, fft_cch, dccch, t, featureSummary, 
                       ifSave=True, savePath=None, filename=None):

    # extract pre/post features
    # -----------------------------------------------------------------------------------------------------
    pre_fs = featureSummary[featureSummary['neuron_name']== preName][['real_FR', 'burstingFraction', 'cv_ISI']].values[0]
    pre_FS = 'FR='+str(round(pre_fs[0],2))+', BF='+str(round(pre_fs[1],2))+', CV='+str(round(pre_fs[2],2))

    post_fs = featureSummary[featureSummary['neuron_name.1']== postName][['firing_rate', 'cv_ISI.1']].values[0]
    post_FS = 'FR='+str(round(post_fs[0],2))+', CV='+str(round(post_fs[1],2))   
    
    
    fig, ax = plt.subplots(3, 3, figsize=(12, 8))

    # first col - ach1
    ax[0, 0].plot(t, ach1, lw=2, c='b'); ax[0, 0].set_title('ach1-'+pre_FS)
    ax[1, 0].plot(ach1_normed, lw=2, c='b'); ax[1, 0].set_title('ach1_normed') 
    # ax[2, 0].plot(freqs_ach1, np.abs(fft_ach1), lw=2, c='b'); ax[2, 0].set_title('ach1_fft')
    ax[2, 0].stem(freqs_ach1, np.abs(fft_ach1), linefmt='b', basefmt='grey'); ax[2, 0].set_title('ach1_fft')

    # second col - ccg
    ax[0, 1].plot(t, cch, lw=3, c='orange'); ax[0, 1].set_title('cch')
    # ax[1, 1].plot(freqs_cch, np.abs(fft_cch), lw=2, c='orange'); ax[1, 1].set_title('cch_fft') 
    ax[1, 1].stem(freqs_cch, np.abs(fft_cch), linefmt='orange', basefmt='grey',); ax[1, 1].set_title('cch_fft') 
    ax[2, 1].plot(t, dccch, lw=3, c='orange'); ax[2, 1].set_title('dcCCH')

    # third col - ach2
    ax[0, 2].plot(t, ach2, lw=2, c='g'); ax[0, 2].set_title('ach2-'+post_FS)
    ax[1, 2].plot(ach2_normed, lw=2, c='g'); ax[1, 2].set_title('ach2_normed') 
    # ax[2, 2].plot(freqs_ach2, np.abs(fft_ach2), lw=2, c='g'); ax[2, 2].set_title('ach2_fft')
    ax[2, 2].stem(freqs_ach2, np.abs(fft_ach2), linefmt='g', basefmt='grey'); ax[2, 2].set_title('ach2_fft')

    plt.tight_layout()
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_CM(CM_plot, numE, numI, eColor='magenta', iColor='cyan', pixelScale=0.3, figsize=(10, 10), ifSave=True, savePath=None, filename=None):

    '''
    plot connectivity matirx
        - color represent synapse type (exc (e2e, e2i) or inh(i2e, i2e))
        - black - no connectivity
        
    
    '''
    numS = CM_plot.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, numS-0.5); ax.set_ylim(-0.5, numS-0.5)
    ax.grid(False); ax.set_frame_on(False); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((-0.5, -0.5), numS, numS, color="black"))
    
    for i in range(numS):
        for j in range(numS):
            if CM_plot[i, j] != 0:  # a connection
                size = CM_plot[i, j] * pixelScale 
                color = eColor if i < numE else iColor
                ax.add_patch(plt.Rectangle( (j-size/2, numS-1-i-size/2), size, size, color=color))
    
    ax.axhline(y=numI-0.5, color='lightgrey', linewidth=2) # dividing rows(upper - exc pre, lower - inh pre)
    ax.axvline(x=numE-0.5, color='lightgrey', linewidth=2) # dividing cols(upper - exc post, lower - inh post)
    
    plt.tight_layout()
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename), bbox_inches="tight")
        plt.close()
    else:
        plt.show()














