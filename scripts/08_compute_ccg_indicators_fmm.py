"""
08_compute_ccg_indicators_fmm.py
python scripts/08_compute_ccg_indicators_fmm.py --help
"""


# Import Packages
#=================================================================================================================
import os
import argparse
import pandas as pd
from Functions import *

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "PointNeuron_Simulation"))

import warnings
warnings.filterwarnings('ignore')


# arguments
#=================================================================================================================
parser = argparse.ArgumentParser("Signal Analysis - indicator")

parser.add_argument('--dataFolderName', default='data', type=str)
parser.add_argument('--base_segFolderName', default='base', type=str)
parser.add_argument('--perb_segFolderName', default='BP', type=str)

parser.add_argument('--dataloaderName', default='TTV_2_3.pkl', type=str)
parser.add_argument('--unseen_dlName', default='oneDL_1.pkl', type=str) 

parser.add_argument('--resultFolderName', default='test', type=str)
parser.add_argument('--modelName', default='cRaw_bestVal_valAccu100.0000_testAccu100.0000.pth', type=str)
parser.add_argument('--indiFolderName', default='indi', type=str)


args = parser.parse_args()

# path & params
#=================================================================================================================
root_path = str(PROJECT_ROOT)
data_path = os.path.join(root_path, args.dataFolderName)
dataloader_path = os.path.join(data_path, 'dataloaders')

result_path = os.path.join(root_path, 'results', args.resultFolderName)
indiResultSave_path = os.path.join(result_path, args.indiFolderName); os.makedirs(indiResultSave_path, exist_ok=True)

bin_size=1; timebins = list(range(-50, 51))


# Indicator for each CCG
#=================================================================================================================
indicator_exc_list = []; indicator_inh_list = []
for fileTest in [args.base_segFolderName, args.perb_segFolderName]:
    try:
        ccg, ws, ls, ss = read_ccg(data_path, f'{fileTest}_CCG.csv') 
        plot_exc = True; plot_inh = True
    
        for nIdx in range(len(ccg)):
            w = ws[nIdx]; sname = ss[nIdx]
            if w > 0:
                if plot_exc:
                    indi = ccg_indicators(ccg[nIdx], connection_type='exc', smoothSigma=0.5, bin_size=1, peak_window_ms=10, timebins=timebins, ifVerbose=False,ifPlot=True,ifSave=True,
                                          savePath=os.path.join(indiResultSave_path, 'plots'), filename=sname+'.png')
                    plot_exc=False
                else:
                    indi = ccg_indicators(ccg[nIdx], connection_type='exc', smoothSigma=0.5, bin_size=1, peak_window_ms=10, ifVerbose=False,ifPlot=False,ifSave=False)
                indi['Weight'] = w; indi['sample']=sname; indi['test']=fileTest; indicator_exc_list.append(indi)
            
            elif w < 0:
                if plot_inh:
                    indi = ccg_indicators(ccg[nIdx], connection_type='inh', smoothSigma=0.5, bin_size=1, peak_window_ms=10, timebins=timebins, ifVerbose=False,ifPlot=True,ifSave=True,
                                          savePath=os.path.join(indiResultSave_path, 'plots'), filename=sname+'.png')
                    plot_inh = False
                else:
                    indi = ccg_indicators(ccg[nIdx], connection_type='inh', smoothSigma=0.5, bin_size=1, peak_window_ms=10, ifVerbose=False,ifPlot=False,ifSave=False)
                indi['Weight'] = w; indi['sample']=sname; indi['test']=fileTest; indicator_inh_list.append(indi)

    except Exception as e:
        print('  -', fileTest, ':', e)

indicatorExc_df=pd.DataFrame(indicator_exc_list)
indicatorInh_df=pd.DataFrame(indicator_inh_list)


# fmm
#=================================================================================================================
predSummary_list = []
fm_mean_cols = ['conv1_fm0_mean', 'conv1_fm1_mean', 'conv1_fm2_mean', 'conv1_fm3_mean', 'conv2_fm0_mean', 'conv2_fm1_mean']

model, optimizer, scheduler, criterion = init_conn_model()
connCNN_Model, _ = load_conn_model(model,optimizer,scheduler,savePath=result_path, filename=args.modelName)

# predict on baseline-test set 
# ------------------------------------------------
train_loader, val_loader, test_loader = load_dataloaders(dataloader_path, args.dataloaderName)
avg_connect_loss, accu, pred_list,confi_list, label_list, w_list, sCInfo_list = evaluateConnCNN_wConfi_model(connCNN_Model, test_loader, criterion)

# feature map mean
conn_fmMean_list = []
conn_fmMean_pre_list = []
conn_fmMean_postBN_list = []
for ccgs, cs, ws, ss in test_loader:
    for m in range(len(ccgs)):
        # pre activation (raw featureMaps)
        fm_conv1, fm_conv2, activation_1, activation_2 = cal_feature_maps(connCNN_Model, ccgs[m])  # connCNN
        conn_fmMean_pre_list.append(fm_conv1.mean(dim=1).tolist() + fm_conv2.mean(dim=1).tolist())

        # postBN (featureMaps after BatchNorm, but before tanh)
        fm_conv1, fm_conv2, activation_1, activation_2 = cal_feature_maps_postBN(connCNN_Model, ccgs[m])  # connCNN
        conn_fmMean_postBN_list.append(fm_conv1.mean(dim=1).tolist() + fm_conv2.mean(dim=1).tolist())

        # post activation (featureMaps after BatchNorm & tanh)
        fm_conv1, fm_conv2, activation_1, activation_2 = cal_feature_maps_post_activation(connCNN_Model, ccgs[m])  # connCNN
        conn_fmMean_list.append(fm_conv1.mean(dim=1).tolist() + fm_conv2.mean(dim=1).tolist())
        
predSummary = pd.DataFrame([label_list, pred_list, confi_list, sCInfo_list,],
                            index=['connLabel', 'predLabel', 'confi', 'sample']).T   
connPre_FMM = pd.DataFrame(np.array(conn_fmMean_pre_list), columns=['pre_conn_'+col for col in fm_mean_cols])
connPost_FMM = pd.DataFrame(np.array(conn_fmMean_list), columns=['post_conn_'+col for col in fm_mean_cols])
connPostBN_FMM = pd.DataFrame(np.array(conn_fmMean_postBN_list), columns=['postBN_conn_'+col for col in fm_mean_cols])
predSummary = pd.concat([predSummary, connPre_FMM, connPost_FMM, connPostBN_FMM], axis=1); predSummary_list.append(predSummary)


# predict on associated perturbations
# ------------------------------------------------
perb_loader = load_oneLoader(dataloader_path, args.unseen_dlName)
avg_connect_loss, accu, pred_list,confi_list, label_list, w_list, sCInfo_list = evaluateConnCNN_wConfi_model(connCNN_Model, perb_loader, criterion)

# feature map mean
conn_fmMean_list = []
conn_fmMean_pre_list = []
conn_fmMean_postBN_list = []
for ccgs, cs, ws, ss in perb_loader:
    for m in range(len(ccgs)):

        # pre activation (raw featureMaps)
        fm_conv1, fm_conv2, activation_1, activation_2 = cal_feature_maps(connCNN_Model, ccgs[m])  # connCNN
        conn_fmMean_pre_list.append(fm_conv1.mean(dim=1).tolist() + fm_conv2.mean(dim=1).tolist())

        # postBN (featureMaps after BatchNorm, but before tanh)
        fm_conv1, fm_conv2, activation_1, activation_2 = cal_feature_maps_postBN(connCNN_Model, ccgs[m])  # connCNN
        conn_fmMean_postBN_list.append(fm_conv1.mean(dim=1).tolist() + fm_conv2.mean(dim=1).tolist())

        # post activation (featureMaps after BatchNorm & tanh)
        fm_conv1, fm_conv2, activation_1, activation_2 = cal_feature_maps_post_activation(connCNN_Model, ccgs[m])
        conn_fmMean_list.append(fm_conv1.mean(dim=1).tolist() + fm_conv2.mean(dim=1).tolist())

predSummary = pd.DataFrame([label_list, pred_list, confi_list, sCInfo_list,],
                            index=['connLabel', 'predLabel', 'confi', 'sample']).T
connPre_FMM = pd.DataFrame(np.array(conn_fmMean_pre_list), columns=['pre_conn_'+col for col in fm_mean_cols])
connPost_FMM = pd.DataFrame(np.array(conn_fmMean_list), columns=['post_conn_'+col for col in fm_mean_cols])
connPostBN_FMM = pd.DataFrame(np.array(conn_fmMean_postBN_list), columns=['postBN_conn_'+col for col in fm_mean_cols])
predSummary = pd.concat([predSummary, connPre_FMM, connPost_FMM, connPostBN_FMM], axis=1); predSummary_list.append(predSummary)

bspredSummary = pd.concat(predSummary_list)



# merge fmm with indi
#=================================================================================================================
pred_indi_exc = pd.merge(bspredSummary, indicatorExc_df, on='sample', how='inner')
pred_indi_exc.to_csv(os.path.join(indiResultSave_path, f'base{args.base_segFolderName}_perb{args.perb_segFolderName}_indi_fmm_exc.csv'))

pred_indi_inh = pd.merge(bspredSummary, indicatorInh_df, on='sample', how='inner')
pred_indi_inh.to_csv(os.path.join(indiResultSave_path, f'base{args.base_segFolderName}_perb{args.perb_segFolderName}_indi_fmm_inh.csv'))

print(f"--Done computing ccg indicators & FMM for baseline dataset {args.base_segFolderName} and perturbation dataset {args.perb_segFolderName} \n")



