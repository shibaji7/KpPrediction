##
#
##

import sys
import os
import datetime as dt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

import database as db
import util

np.random.seed(7)

def build_all_classifiers():
    # Dataset
    _xparams,X,y = db.load_data_for_deterministic_bin_clf()
    rus = RandomUnderSampler(return_indices=True)
    X_resampled, y_resampled, idx_resampled = rus.fit_sample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=1.0/3.0, random_state=42)

    # Initialize metrix
    CLF = util.get_classifiers()
    roc_eval_details = {}
    feature_importance = {}
    _C2x2 = {}
    for I,clfs in enumerate(CLF):
        gname = clfs["name"]
        roc_eval_details[gname] = {}
        for clf,name in clfs["methods"]:
            print("--> Running '%s' model" % name)
            clf.fit(X_train, y_train)
            if hasattr(clf,"feature_importances_"):
                feature_importance[name] = dict(zip(_xparams,clf.feature_importances_))
                pass
            if hasattr(clf, "predict_proba") or hasattr(clf, "decision_function"):
                roc_eval_details[gname][name] = {}
                roc_eval_details[gname][name]["y_score"], roc_eval_details[gname][name]["fpr"], roc_eval_details[gname][name]["tpr"], roc_eval_details[gname][name]["roc_auc"] = util.get_roc_details(clf, X_test, y_test)
                roc_eval_details[gname][name]["c"] = np.random.rand(3,1).ravel().tolist()
                pass
            _C2x2[name] = util.validate_model_matrices(clf, X_test, y_test)
            pass
        pass
    util.plot_deterministic_roc_curves(roc_eval_details)
    for name in _C2x2.keys():
        print("Running '%s' model" % name)
        print("==================")
        _C2x2[name].summary(verbose=True)
        print("**************************************************")
        print("\n\n")
        pass
    return






if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0: print "Invalid call sequence!! python jobutil.py {1/2/3...}"
    else:
        ctx = int(args[0])
        if ctx == 1: build_all_classifiers()
        else: print "Invalid option!!"
        pass
    pass
    
