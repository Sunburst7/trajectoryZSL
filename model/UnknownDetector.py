import torch
import torch.nn as nn
import os
import numpy as np
import math
import pickle
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import dtw_path_from_metric
from scipy.signal import correlate
import sys

class UnknownDetector(nn.Module):
    def __init__(self, cfg, STD_COEF_1=1.0, STD_COEF_2=1.0):
        super().__init__()
        self.cfg = cfg
        self.num_class = cfg.dataset.num_class - 1 # known classes
        
        self.bc_list = []
        for c in range(self.num_class):
            bc_data_path = os.path.join(self.cfg.dataset.root_data_path, "barycenters", f"bc_{c}.npy")
            bc = np.load(bc_data_path)
            self.bc_list.append(bc)

        with open(os.path.join(cfg.dataset.root_data_path, 'all_dist.pkl'), 'rb') as f1:
            self.distances_all =  pickle.load(f1)

        with open(os.path.join(cfg.dataset.root_data_path, 'all_cc.pkl'), 'rb') as f2:
            self.cc_all =  pickle.load(f2)

        self.top_distances_for_class = []
        for _, dist_for_class in self.distances_all.items():
            mean = np.median(dist_for_class)
            std = np.std(dist_for_class)
            self.top_distances_for_class.append(mean + STD_COEF_1*std)

        self.bottom_cc_for_class = []
        for _, cc_for_class in self.cc_all.items():
            mean = np.median(cc_for_class, axis=0)
            std = np.std(cc_for_class, axis=0)
            self.bottom_cc_for_class.append(mean - STD_COEF_2*std)
       
    
    def calc_distances_to_each_bc(self, x, bc_list):
        distances = []
        for bc in bc_list:
            dist = dtw_path_from_metric(x, bc, metric='sqeuclidean')[1]
            distances.append(math.log(dist))
        return np.asarray(distances)
    
    def unknown_detector(self, x):
        """Returns True if the sample is further away (DTW distance) from each class than the most extreme train samples"""
        
        bc_distances = self.calc_distances_to_each_bc(x, self.bc_list)
        #print(bc_distances)
        if all(bc_distances > self.top_distances_for_class):
            return True
        else:
            #print(np.where(bc_distances < top_distances_for_class))
            return False

    def cc_unknown_detector(self, x):
        """Returns True if the sample is not cross-correlated with any of the known classes"""

        ccs = []
        for i in range(self.num_class):
            __ = []
            for j in range(x.shape[1]):
                cc = np.max(correlate(x[:, j], self.bc_list[i][:, j]))
                __.append(cc)
            ccs.append(__)
        ccs = np.asarray(ccs)
        
        cnt = 0
        flags = (ccs > self.bottom_cc_for_class)
        # print(flags)
        for row in flags:
            if np.isin(False, row):
                cnt += 1

        if cnt == self.num_class:
            return True
        else:
            return False
        
    def forward(self, x, preds):
        """Modify the prediction to be unknown if one of the detectors return True"""
        x = torch.permute(x, (0, 2, 1))
        x = x.cpu().detach().numpy()
        for n in range(len(x)):
            if self.cc_unknown_detector(x[n]) or self.unknown_detector(x[n]):
                preds[n] = self.num_class
        return preds
    

def split_data_by_class(x, y):
    """
    Returns sorted splits for each class
    """
    # Sort by the target values so that it can be split for each class 
    sorted_idxs = np.argsort(y, axis=0)
    x = x[sorted_idxs]
    y = y[sorted_idxs]
    # Group by target, i.e. one split for each class
    splits = np.split(x, np.unique(y, return_index = True)[1][1:])
    print(f"Number of splits: {len(splits)}")
    return splits

class UnknownDetectorWeibull(nn.Module):
    def __init__(self, cfg, x_correct, y_correct, av_correct):
        self.cfg = cfg
        self.x_correct = x_correct
        self.y_correct = y_correct
        self.n_classes = cfg.dataset.num_class - 1
        # OpenMax
        sorted_idxs = torch.argsort(y_correct, axis=0)
        x = x_correct[sorted_idxs]
        y = y_correct[sorted_idxs]
        av = av_correct[sorted_idxs]
        splits = torch.split(x, torch.unique(y, return_index = True)[1][1:])
        av_splits = torch.split(av, torch.unique(y, return_index = True)[1][1:])
        if len(splits) != self.n_classes:
            print(f'There are {self.n_classes - len(splits)} classes with zero correctly classified samples')
            print('It is not possible to fit Weibull model for each class')
            sys.exit()
        mav = torch.mean(av, axis=0)


        # Calculate distances to MAVs
        def calc_distances(activations, mav):
            """
            Returns the distances between the activations of correctly classified samples and MAV of the given class
            """
            distances = np.empty(len(activations))
            for i in range(len(activations)):
                distances[i] = np.linalg.norm(activations[i] - mav)
            return distances

        def get_top_distances(distances, ratio=0.1):
            """
            Returns the top r% largest distances of the given array
            """
            sorted_dists = np.sort(distances)
            return sorted_dists[-round(len(distances)*ratio):]

    #     top_distances_all = []
    #     for c_i in range(self.n_classes):
    #         # distances between each sample from the given class and its MAV
    #         distances = calc_distances(av_list[c_i], mav_list[c_i])
    #         top_dists = get_top_distances(distances)
    #         top_distances_all.append(top_dists)


    # def fit_weibull(distances_all, n_classes):
    #     """
    #     Returns one Weibull model (set of parameters) for each class
    #     """
    #     weibull_models = []
    #     for i in range(n_classes):
    #         shape, loc, scale = weibull_min.fit(distances_all[i])
    #         weibull_models.append([shape, loc, scale])

    #     # Save Weibull models
    #     weibull_path = os.path.join(MODELS_PATH, f'{dataset}/{dataset}_Weibull.pkl')
    #     with open(weibull_path, 'wb') as f:
    #         pickle.dump(weibull_models, f)

    #     return weibull_models


    # weibull_models = fit_weibull(top_distances_all, n_classes)