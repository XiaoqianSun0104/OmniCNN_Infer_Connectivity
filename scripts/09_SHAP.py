"""
09_SHAP.py
python scripts/09_SHAP.py --help
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

import shap
from functools import reduce
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# arguments
#=================================================================================================================
parser = argparse.ArgumentParser("Signal Analysis - indicator")

parser.add_argument('--base_segFolderName', default='base', type=str)
parser.add_argument('--perb_segFolderName', default='BP', type=str)

parser.add_argument('--resultFolderName', default='test', type=str)
parser.add_argument('--indiFolderName', default='indi', type=str)

args = parser.parse_args()

# path & params
#=================================================================================================================
root_path = str(PROJECT_ROOT)

result_path = os.path.join(root_path, 'results', args.resultFolderName)
indiResultSave_path = os.path.join(result_path, args.indiFolderName)

indicators_exc = ['peak_height', 'peak_to_noise', 'norm_entropy', 'kl_divergence_window']
indicators_inh = ['dip_depth', 'dip_to_noise', 'norm_entropy', 'kl_divergence_window']

pred_indi_exc = pd.read_csv(os.path.join(indiResultSave_path, f'base{args.base_segFolderName}_perb{args.perb_segFolderName}_indi_fmm_exc.csv'))
pred_indi_inh = pd.read_csv(os.path.join(indiResultSave_path, f'base{args.base_segFolderName}_perb{args.perb_segFolderName}_indi_fmm_inh.csv'))


# EXC
#=================================================================================================================
pred_indi_exc['hit'] = (pred_indi_exc['connLabel'] == pred_indi_exc['predLabel']).astype(int)
counts = pred_indi_exc['hit'].value_counts(); 

if counts.get(0, 0) > 5:
    print(f"--Exc 0-miss-{counts.get(0, 0)} |", f"1-hit-{counts.get(1, 0)}")
    X = pred_indi_exc[indicators_exc]
    y = pred_indi_exc['hit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    
    # RF ------------------------------------------------------------------------------------------
    RF_Model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    RF_Model.fit(X_train, y_train); RF_yPred = RF_Model.predict(X_test)
    RF_report = pd.DataFrame(classification_report(y_test, RF_yPred, output_dict=True)).transpose().reset_index()
    RF_FI = pd.DataFrame([X.columns, RF_Model.feature_importances_], index=['feature', 'importance']).T
    RF_Summary = pd.concat([RF_report, RF_FI], axis=1); 
    
    
    # SHAP explainer --------------------------------------------------------------------------------
    RF_explainer = shap.TreeExplainer(RF_Model); shap_values = RF_explainer.shap_values(X)

    # RF/SHPA FI Summary ----------------------------
    mean_shap = pd.DataFrame({'feature': X.columns, 'MeanAbsSHAP': np.abs(shap_values[:, :, 1]).mean(axis=0)})
    mean_shap['MeanAbsSHAPProb'] = mean_shap[['MeanAbsSHAP']]/mean_shap[['MeanAbsSHAP']].sum(); 
    var_shap = pd.DataFrame({'feature': X.columns, 'varanceSHAP': shap_values[:, :, 1].var(axis=0)})
    RF_SHAP_summary = reduce(lambda left, right: pd.merge(left, right, on='feature'), [RF_Summary, mean_shap, var_shap])
    RF_SHAP_summary['hitRate'] = pred_indi_exc['hit'].mean() 
    RF_SHAP_summary.to_csv(os.path.join(indiResultSave_path, 'RF_SHAP_Exc.csv'))
    
    # sample-wise shap/indi values ------------------
    if isinstance(shap_values, list): 
        shap_values = shap_values[1]
    elif shap_values.ndim == 3: 
        shap_values = shap_values[:, :, 1]
    shap_cols = {f'shap_{f}': shap_values[:, X.columns.get_loc(f)] for f in indicators_exc}
    raw_cols  = {f: pred_indi_exc[f].values for f in indicators_exc}
    base_cols = {'connLabel':   pred_indi_exc['connLabel'].values,
                    'predLabel': pred_indi_exc['predLabel'].values,
                    'sample':         pred_indi_exc['sample'].values,
                    'test':        pred_indi_exc['test'].values}
    shap_all_contri = pd.DataFrame({**base_cols, **raw_cols, **shap_cols})
    shap_all_contri['hit/miss'] = (shap_all_contri['connLabel'] == shap_all_contri['predLabel']).map({True: 'hit', False: 'miss'})
    shap_all_contri.to_excel(os.path.join(indiResultSave_path, 'sampleIndiSHAP_Exc.xlsx'))
else:
    print('--Exc skipping:', f"0-miss-{counts.get(0, 0)} |", f"1-hit-{counts.get(1, 0)}")



# INH
#=================================================================================================================
pred_indi_inh['hit'] = (pred_indi_inh['connLabel'] == pred_indi_inh['predLabel']).astype(int)
counts = pred_indi_inh['hit'].value_counts(); 

if counts.get(0, 0) > 5:
    print(f"--Inh 0-miss-{counts.get(0, 0)} |", f"1-hit-{counts.get(1, 0)}")
    X = pred_indi_inh[indicators_inh]
    y = pred_indi_inh['hit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    
    # RF ------------------------------------------------------------------------------------------
    RF_Model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    RF_Model.fit(X_train, y_train); RF_yPred = RF_Model.predict(X_test)
    RF_report = pd.DataFrame(classification_report(y_test, RF_yPred, output_dict=True)).transpose().reset_index()
    RF_FI = pd.DataFrame([X.columns, RF_Model.feature_importances_], index=['feature', 'importance']).T
    RF_Summary = pd.concat([RF_report, RF_FI], axis=1); 
    
    
    # SHAP explainer --------------------------------------------------------------------------------
    RF_explainer = shap.TreeExplainer(RF_Model); shap_values = RF_explainer.shap_values(X)

    # RF/SHPA FI Summary ----------------------------
    mean_shap = pd.DataFrame({'feature': X.columns, 'MeanAbsSHAP': np.abs(shap_values[:, :, 1]).mean(axis=0)})
    mean_shap['MeanAbsSHAPProb'] = mean_shap[['MeanAbsSHAP']]/mean_shap[['MeanAbsSHAP']].sum(); 
    var_shap = pd.DataFrame({'feature': X.columns, 'varanceSHAP': shap_values[:, :, 1].var(axis=0)})
    RF_SHAP_summary = reduce(lambda left, right: pd.merge(left, right, on='feature'), [RF_Summary, mean_shap, var_shap])
    RF_SHAP_summary['hitRate'] = pred_indi_inh['hit'].mean() 
    RF_SHAP_summary.to_csv(os.path.join(indiResultSave_path, 'RF_SHAP_Inhc.csv'))
    
    # sample-wise shap/indi values ------------------
    if isinstance(shap_values, list): 
        shap_values = shap_values[1]
    elif shap_values.ndim == 3: 
        shap_values = shap_values[:, :, 1]
    shap_cols = {f'shap_{f}': shap_values[:, X.columns.get_loc(f)] for f in indicators_inh}
    raw_cols  = {f: pred_indi_inh[f].values for f in indicators_inh}
    base_cols = {'connLabel':   pred_indi_inh['connLabel'].values,
                    'predLabel': pred_indi_inh['predLabel'].values,
                    'sample':         pred_indi_inh['sample'].values,
                    'test':        pred_indi_inh['test'].values}
    shap_all_contri = pd.DataFrame({**base_cols, **raw_cols, **shap_cols})
    shap_all_contri['hit/miss'] = (shap_all_contri['connLabel'] == shap_all_contri['predLabel']).map({True: 'hit', False: 'miss'})
    shap_all_contri.to_excel(os.path.join(indiResultSave_path, 'sampleIndiSHAP_Inh.xlsx'))
else:
    print('--Inh skipping:', f"0-miss-{counts.get(0, 0)} |", f"1-hit-{counts.get(1, 0)}")

print('--Done.\n')

