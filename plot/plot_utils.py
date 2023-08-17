import numpy as np
from scipy import stats

def compute_mean_ci(data, confidence_level=0.95):
    """
    Compute the mean and confidence interval of a vector using a specified confidence level.
    
    Parameters:
        data (numpy array or list): Input data as a vector.
        confidence_level (float): Confidence level to compute the interval. Default is 0.95.
    
    Returns:
        tuple: A tuple containing the mean and confidence interval as (mean, (lower_bound, upper_bound)).
    """
    sample_mean = np.mean(data)
    n = len(data)
    std_err = stats.sem(data)
    margin_of_error = std_err * stats.t.ppf((1 + confidence_level) / 2, n - 1)
    # ci = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    return sample_mean, margin_of_error


