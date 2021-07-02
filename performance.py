from collections import OrderedDict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_fscore_support

def get_performance(y_trues, y_preds):
    fpr, tpr, t = roc_curve(y_trues, y_preds, pos_label=1)
    roc_score = auc(fpr, tpr)
    
    #Threshold
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(t, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    threshold = roc_t['threshold']
    threshold = list(threshold)[0]
    
    y_preds = [1 if ele >= threshold else 0 for ele in y_preds] 
    
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_trues, y_preds, average="binary", pos_label=1)
    #### conf_matrix = [["true_normal", "false_abnormal"], ["false_normal", "true_abnormal"]]     
    conf_matrix = confusion_matrix(y_trues, y_preds)
    performance = OrderedDict([ ('AUC', roc_score), ('precision', precision),
                                ("recall", recall), ("F1_Score", f1_score), ("conf_matrix", conf_matrix),
                                ("threshold", threshold)])
                                
    return performance, t

def get_values_for_pr_curve(y_trues, y_preds, thresholds):
    precisions = []
    recalls = []
    tn_counts = []
    fp_counts = []
    fn_counts = []
    tp_counts = []
    for threshold in thresholds:
        y_preds_new = [1 if ele >= threshold else 0 for ele in y_preds] 
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds_new).ravel()
        if len(set(y_preds_new)) == 1:
            print("y_preds_new did only contain the element {}... Continuing with next iteration!".format(y_preds_new[0]))
            continue
        
        precision, recall, _, _ = precision_recall_fscore_support(y_trues, y_preds_new, average="binary", pos_label=1)
        precisions.append(precision)
        recalls.append(recall)
        tn_counts.append(tn)
        fp_counts.append(fp)
        fn_counts.append(fn)
        tp_counts.append(tp)
        
        
    
    return np.array(tp_counts), np.array(fp_counts), np.array(tn_counts), np.array(fn_counts), np.array(precisions), np.array(recalls), len(thresholds)