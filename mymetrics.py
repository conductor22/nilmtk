import numpy as np
import pandas as pd
import math
from warnings import warn

'''
For the functions to work with dataframes containing multiple columns, 
specify column in each dataframe operation
'''

def error_in_assigned_energy(predictions, ground_truth_dict):
    """Compute error in assigned energy.

    .. math::
        error^{(n)} = 
        \\left | \\sum_t y^{(n)}_t - \\sum_t \\hat{y}^{(n)}_t \\right |

    Parameters
    ----------
    predictions, ground_truth : list of pd.DataFrame

    Returns
    -------
    errors : pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the absolute error in assigned energy for that appliance,
        in kWh.
    """
    ground_truth = list(ground_truth_dict.values())
    errors = []
    for pred_df, gt_df in zip(predictions, ground_truth):
        predicted_energy = pred_df.sum()
        predicted_energy = np.float64(predicted_energy)
        ground_truth_energy = gt_df.sum()
        error = np.abs(ground_truth_energy - predicted_energy)
        errors.append(error)
    return errors

def fraction_energy_assigned_correctly(predictions, ground_truth_dict):
    '''
    sagt irgendwie gar nichts aus
    '''
    ground_truth = list(ground_truth_dict.values())
    total_ground_truth_energy = sum(df.sum().sum() for df in ground_truth)
    total_predicted_energy = sum(df.sum().sum() for df in predictions)

    fraction = 0
    for pred_df, gt_df in zip(predictions, ground_truth):
        predicted_energy = pred_df.sum().sum()
        predicted_energy = np.float64(predicted_energy)
        ground_truth_energy = gt_df.sum().sum()
        
        fraction += min(ground_truth_energy / total_ground_truth_energy,
                        predicted_energy / total_predicted_energy)
    return fraction

def normalized_mean_absolute_error_power(predictions, ground_truth_dict):

    ground_truth = list(ground_truth_dict.values())
    nae = []
    for i, (pred_df, gt_df) in enumerate(zip(predictions, ground_truth)):
        total_abs_diff = 0.0
        sum_of_ground_truth_power = 0.0
        aligned_df = pd.concat([pred_df, gt_df], axis=1, join='inner').dropna()
        pred_col = aligned_df.iloc[:, 0]
        gt_col = aligned_df.iloc[:, 1]
        diff = pred_col - gt_col
        total_abs_diff += sum(abs(diff.dropna()))
        sum_of_ground_truth_power += aligned_df.iloc[:, 1].sum()

        nae.append(total_abs_diff / sum_of_ground_truth_power)

    return nae

def mean_absolute_error_power(predictions, ground_truth_dict):

    ground_truth = list(ground_truth_dict.values())
    mae = []
    for i, (pred_df, gt_df) in enumerate(zip(predictions, ground_truth)):
        total_abs_diff = 0.0
        # sum_of_ground_truth_power = 0.0
        aligned_df = pd.concat([pred_df, gt_df], axis=1, join='inner').dropna()
        pred_col = aligned_df.iloc[:, 0]
        gt_col = aligned_df.iloc[:, 1]
        
        diff = pred_col - gt_col
        diff.dropna(inplace=True)

        total_abs_diff = sum(abs(diff.dropna()))

        mae.append(total_abs_diff / len(diff))

    return mae

def rms_error_power(predictions, ground_truth_dict):
    '''Compute RMS error in assigned power
    
    .. math::
            error^{(n)} = \\sqrt{ \\frac{1}{T} \\sum_t{ \\left ( y_t - \\hat{y}_t \\right )^2 } }

    Parameters
    ----------
    predictions, ground_truth : list of pd.DataFrame
        Each DataFrame corresponds to the predictions or ground truth for a meter.

    Returns
    -------
    error : pd.Series
        Each index is an integer representing the meter instance.
        Each value is the RMS error in predicted power for that appliance.
    '''
    ground_truth = list(ground_truth_dict.values())
    error = []

    for i, (pred_df, gt_df) in enumerate(zip(predictions, ground_truth)):
        sum_of_squared_diff = 0.0
        n_samples = 0
        aligned_df = pd.concat([pred_df, gt_df], axis=1, join='inner').dropna()


        pred_col = aligned_df.iloc[:, 0]
        gt_col = aligned_df.iloc[:, 1]
        diff = pred_col - gt_col
        diff.dropna(inplace=True)

        squared_diff = (diff ** 2)
        mean_squared_diff = squared_diff.sum() / len(diff)
        rmse = math.sqrt(mean_squared_diff)

        error.append(rmse)
        # error.append(math.sqrt(sum_of_squared_diff / n_samples))

    return error

def f1_score(predictions, ground_truth_dict):
    '''Compute F1 scores.

    .. math::
        F_{score}^{(n)} = \\frac
            {2 * Precision * Recall}
            {Precision + Recall}

    Parameters
    ----------
    predictions, ground_truth : list of pd.DataFrame
        Each DataFrame corresponds to the predictions or ground truth for a meter.

    Returns
    -------
    f1_scores : pd.Series
        Each index is an integer representing the meter instance.
        Each value is the F1 score for that appliance. If there are multiple
        chunks then the value is the weighted mean of the F1 score for
        each chunk.
    '''
    from sklearn.metrics import f1_score as sklearn_f1_score
    f1_scores = []

    threshold = 5

    ground_truth = list(ground_truth_dict.values())
    for i, (pred_df, gt_df) in enumerate(zip(predictions, ground_truth)):
        pred_df_binary = (pred_df > threshold).astype(int)
        gt_df_binary = (gt_df > threshold).astype(int)
        aligned_df = pd.concat([pred_df_binary, gt_df_binary], axis=1, join='inner').dropna()
        

   

        score = sklearn_f1_score(aligned_df.iloc[:, 0], aligned_df.iloc[:, 1])

        f1_scores.append(score)

    return f1_scores
    
    tolerance = 0.1
    ground_truth = list(ground_truth_dict.values())
    for i, (pred_df, gt_df) in enumerate(zip(predictions, ground_truth)):
        scores_for_meter = pd.DataFrame(columns=['score', 'num_samples'])
        aligned_df = pd.concat([pred_df, gt_df], axis=1, join='inner').dropna()
        aligned_df = aligned_df.astype(int)

        lower_bound = gt_df * (1 - tolerance)
        upper_bound = gt_df * (1 + tolerance)
        lower_bound.columns = pred_df.columns
        upper_bound.columns = pred_df.columns
        gt_df.columns = pred_df.columns
        # aligned_df = aligned_df.astype(int)

        y_pred = ((pred_df >= lower_bound) & (pred_df <= upper_bound)).astype(int)
        y_true = (gt_df > 0).astype(int)
        score = sklearn_f1_score(y_true, y_pred)
        # scores_for_meter = scores_for_meter.append(
        #     {'score': score, 'num_samples': len(aligned_pred_df)},
        #     ignore_index=True)
        f1_scores.append(score)

    return f1_scores