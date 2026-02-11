import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class SimpleBayesClassifier:

    def __init__(self, n_pos, n_neg):
        
        """
        Initializes the SimpleBayesClassifier with prior probabilities.

        Parameters:
        n_pos (int): The number of positive samples.
        n_neg (int): The number of negative samples.
        
        Returns:
        None: This method does not return anything as it is a constructor.
        """

        self.n_pos = n_pos
        self.n_neg = n_neg
        self.prior_pos = n_pos/(n_pos + n_neg)
        self.prior_neg = n_neg/(n_pos + n_neg)

    def fit_params(self, x, y, n_bins = 10):

        """
        Computes histogram-based parameters for each feature in the dataset.

        Parameters:
        x (np.ndarray): The feature matrix, where rows are samples and columns are features.
        y (np.ndarray): The target array, where each element corresponds to the label of a sample.
        n_bins (int): Number of bins to use for histogram calculation.

        Returns:
        (stay_params, leave_params): A tuple containing two lists of tuples, 
        one for 'stay' parameters and one for 'leave' parameters.
        Each tuple in the list contains the bins and edges of the histogram for a feature.
        """

        self.stay_params = [(None, None) for _ in range(x.shape[1])]
        self.leave_params = [(None, None) for _ in range(x.shape[1])]

        # INSERT CODE HERE
        for idx in range(x.shape[1]):
            x_stay = x[y==0, idx]
            x_stay = x_stay[~np.isnan(x_stay)]
            
            x_leave = x[y==1, idx]
            x_leave = x_leave[~np.isnan(x_leave)]

            for data, param_list in [(x_stay, self.stay_params), (x_leave, self.leave_params)]:
                bins, edges = np.histogram(data, bins=n_bins)
                if len(data) == 0:
                    param_list[idx] = (np.ones(n_bins) / n_bins, edges)
                    continue
                edges = edges.astype(float)
                edges[0] = float('-inf')
                edges[-1] = float('inf')
                alpha = 1
                p = (bins + alpha)/(np.sum(bins) + alpha * n_bins)
                param_list[idx] = (p, edges)
        
        return self.stay_params, self.leave_params

    def predict(self, x, thresh = 0):

        """
        Predicts the class labels for the given samples using the non-parametric model.

        Parameters:
        x (np.ndarray): The feature matrix for which predictions are to be made.
        thresh (float): The threshold for log probability to decide between classes.

        Returns:
        result (list): A list of predicted class labels (0 or 1) for each sample in the feature matrix.
        """

        y_pred = []

        # INSERT CODE HERE
        for sample in x:
            log_p_stay = np.log(self.prior_neg)
            log_p_leave = np.log(self.prior_pos)

            for idx, val in enumerate(sample):
                if np.isnan(val):
                    continue
                p_stay, edges_stay = self.stay_params[idx]
                idx_stay = np.digitize(val, edges_stay) - 1
                idx_stay = np.clip(idx_stay, 0, len(p_stay) - 1)
                log_p_stay += np.log(p_stay[idx_stay])

                p_leave, edges_leave = self.leave_params[idx]
                idx_leave = np.digitize(val, edges_leave) - 1
                idx_leave = np.clip(idx_leave, 0, len(p_leave) - 1)
                log_p_leave += np.log(p_leave[idx_leave])
                
            y_pred.append(1 if (log_p_leave - log_p_stay) > thresh else 0)
        return y_pred
    
    def fit_gaussian_params(self, x, y):

        """
        Computes mean and standard deviation for each feature in the dataset.

        Parameters:
        x (np.ndarray): The feature matrix, where rows are samples and columns are features.
        y (np.ndarray): The target array, where each element corresponds to the label of a sample.

        Returns:
        (gaussian_stay_params, gaussian_leave_params): A tuple containing two lists of tuples,
        one for 'stay' parameters and one for 'leave' parameters.
        Each tuple in the list contains the mean and standard deviation for a feature.
        """

        self.gaussian_stay_params = [(0, 0) for _ in range(x.shape[1])]
        self.gaussian_leave_params = [(0, 0) for _ in range(x.shape[1])]

        # INSERT CODE HERE
        for idx in range(x.shape[1]):
            x_stay = x[y==0, idx]
            x_stay = x_stay[~np.isnan(x_stay)]
            
            x_leave = x[y==1, idx]
            x_leave = x_leave[~np.isnan(x_leave)]

            self.gaussian_stay_params[idx] = (np.mean(x_stay), np.std(x_stay))
            self.gaussian_leave_params[idx] = (np.mean(x_leave), np.std(x_leave))
        
        return self.gaussian_stay_params, self.gaussian_leave_params
    
    def gaussian_predict(self, x, thresh = 0):

        """
        Predicts the class labels for the given samples using the parametric model.

        Parameters:
        x (np.ndarray): The feature matrix for which predictions are to be made.
        thresh (float): The threshold for log probability to decide between classes.

        Returns:
        result (list): A list of predicted class labels (0 or 1) for each sample in the feature matrix.
        """

        y_pred = []

        # INSERT CODE HERE
        for sample in x:
            log_p_stay = np.log(self.prior_neg)
            log_p_leave = np.log(self.prior_pos)

            for idx, val in enumerate(sample):
                m_stay, s_stay = self.gaussian_stay_params[idx]
                m_leave, s_leave = self.gaussian_leave_params[idx]

                log_p_stay += stats.norm.logpdf(val, m_stay, s_stay)
                log_p_leave += stats.norm.logpdf(val, m_leave, s_leave)

            y_pred.append(1 if (log_p_leave - log_p_stay) > thresh else 0)
        return y_pred