import numpy as np
import pandas as pd
from sklearn.utils import resample 
import scipy.stats
from pandas.api.types import is_numeric_dtype

import glob
import sys
sys.path.append('../..')


''' Bootstrap and Confidence Intervals '''
def compute_cis(data, confidence_level=0.05):
    """
    FUNCTION: compute_cis
    ------------------------------------------------------
    Given a Pandas dataframe of (n, labels), return another
    Pandas dataframe that is (3, labels). 
    
    Each row is lower bound, mean, upper bound of a confidence 
    interval with `confidence`. 
    
    Args: 
        * data - Pandas Dataframe, of shape (num_bootstrap_samples, num_labels)
        * confidence_level (optional) - confidence level of interval
        
    Returns: 
        * Pandas Dataframe, of shape (3, labels), representing mean, lower, upper
    """
    data_columns = list(data)
    intervals = []
    for i in data_columns: 
        series = data[i]
        sorted_perfs = series.sort_values()
        lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
        upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
        lower = sorted_perfs.iloc[lower_index].round(4)
        upper = sorted_perfs.iloc[upper_index].round(4)
        mean = round(sorted_perfs.mean(), 4)
        interval = pd.DataFrame({i : [mean, lower, upper]})
        intervals.append(interval)
    intervals_df = pd.concat(intervals, axis=1)
    intervals_df.index = ['mean', 'lower', 'upper']
    return intervals_df

def bootstrap_(y_pred, n_samples=1000): 

    np.random.seed(97)
    y_pred # (500, n_total_labels)
    
    idx = np.arange(len(y_pred))
    
    boot_stats = []
    for i in range(n_samples): 
        sample = resample(idx, replace=True, random_state=i)
        y_pred_ = y_pred[sample]
        boot_stats.append(pd.DataFrame([[np.mean(y_pred_)]]))

    boot_stats = pd.concat(boot_stats) # pandas array of evaluations for each sample
    return boot_stats, compute_cis(boot_stats)

def bootstrap_equity_scaled_perf(identity_wise_perf, overall_perf, no_of_attr, alpha=1., col=0, n_samples=1000): 

    np.random.seed(97)
    
    boot_stats = []
    
    for i in range(n_samples): 

        es_perf = 0
        tmp = 0
        
        for j in range(no_of_attr):
            one_attr_perf_list = identity_wise_perf[j][:,col]
            if one_attr_perf_list is None:
                continue

            idx = np.arange(len(one_attr_perf_list))
            sample = resample(idx, replace=True, random_state=i)
            one_attr_perf_list_ = one_attr_perf_list[sample]

            identity_perf = np.mean(one_attr_perf_list_, axis=0)
            tmp += np.abs(identity_perf-overall_perf)
        
        es_perf = (overall_perf / (alpha*tmp + 1))
        boot_stats.append(pd.DataFrame([[es_perf]]))

    boot_stats = pd.concat(boot_stats) # pandas array of evaluations for each sample
    return boot_stats, compute_cis(boot_stats)

def equity_scaled_perf(identity_wise_perf, overall_perf, no_of_attr, alpha=1.):
    es_perf = 0
    tmp = 0
    
    for i in range(no_of_attr):
        one_attr_perf_list = identity_wise_perf[i]
        if one_attr_perf_list is None:
            continue
        identity_perf = np.mean(one_attr_perf_list, axis=0)
        tmp += np.abs(identity_perf-overall_perf)
    
    es_perf = (overall_perf / (alpha*tmp + 1))
    
    return es_perf

def main():
    
    vis_only = False
    root_runs = 'OUTDIR/3_output/18.1_dMOE/'
    resulst_list = glob.glob("%s/**/result*.csv"%root_runs, recursive=True)

    resulst_list.sort()

    resulst_list_ = []
    for idx, result in enumerate(resulst_list):
        resulst_list_.append(result)
    resulst_list = resulst_list_

    print(resulst_list)

    for flag in ['age', 'race']:
        if flag == 'age':
            with open('%smean_dMoE_rebuttal.csv'%root_runs, 'a', newline='\n') as csvfile:
                csvfile.write("\nmethod, es_dice, dice, 0, , 1, , 2, , 3, , 4, , cnt, , , ,") 
            flag_list = range(0, 5)
        elif flag == 'race':
            with open('%smean_dMoE_rebuttal.csv'%root_runs, 'a', newline='\n') as csvfile:
                csvfile.write("\nmethod, es_dice, dice, 0, , 1, , 2, , cnt, , , ,") 
            flag_list = range(0, 3)

        report_save = []

        for idx, result in enumerate(resulst_list):

            if result.find(flag) < 0:
                continue

            report = pd.read_csv(result) 

            identity_wise_perf, no_of_attr = [], []

            # flag
            for id_flag, attr in enumerate(flag_list):

                metric_list = report[report['Attr'] == attr]['Dice'].values
                identity_wise_perf.append(metric_list)
                metric_list = report[report['Attr'] == attr]['IoU'].values
                identity_wise_perf.append(metric_list)

            if len(identity_wise_perf[1]) == 0:
                continue

            with open('%smean_dMoE_rebuttal.csv'%root_runs, 'a', newline='\n') as csvfile:
                csvfile.write("\n%s, "%(result.replace(root_runs,'')))

            cat_perf = [perf for perf in identity_wise_perf if perf is not None]
            cat_perf = np.concatenate(cat_perf, axis=0)

            for col in range(cat_perf.shape[1]):
                _, boot_stats = bootstrap_(cat_perf[:,col])
                _, boot_stats_essp = bootstrap_equity_scaled_perf(identity_wise_perf, boot_stats.values[0,0], len(flag_list), alpha=1., col=col)
                with open('%smean_dMoE_rebuttal.csv'%root_runs, 'a', newline='\n') as csvfile:
                    csvfile.write("{%.3f (%.3f-%.3f)} & "%(boot_stats_essp.values[0,0],boot_stats_essp.values[1,0],boot_stats_essp.values[2,0]))
                    csvfile.write("{%.3f (%.3f-%.3f)} & "%(boot_stats.values[0,0],boot_stats.values[1,0],boot_stats.values[2,0]))
                
            with open('%smean_dMoE_rebuttal.csv'%root_runs, 'a', newline='\n') as csvfile:
                for flag_idx in range(len(flag_list)):
                    if identity_wise_perf[flag_idx] is None:
                        csvfile.write(" & & ")
                        continue
                    metric = np.mean(identity_wise_perf[flag_idx], axis=0)
                    [csvfile.write("{%.3f} & "%(met)) for met in metric] 
                [csvfile.write(", n=%d"%(len(identity_wise_perf[flag_idx]))) for flag_idx in range(len(flag_list))]

if __name__ == "__main__":
    main()