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
import models as M

np.random.seed(7)

def build_all_classifiers(goes):
    # Dataset
    if goes == "1": _xparams,X,y = db.load_data_for_deterministic_bin_clf()
    else: _xparams,X,y = db.load_data_with_goes_for_deterministic_bin_clf()
    print(X)
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
    util.plot_deterministic_roc_curves(roc_eval_details, goes)
    for name in _C2x2.keys():
        print("Running '%s' model" % name)
        print("==================")
        _C2x2[name].summary(verbose=True)
        print("**************************************************")
        print("\n\n")
        pass
    return

def build_all_regressor():
    regs = ["dummy","regression","elasticnet","bayesianridge","knn","dtree","etree","ada","bagging",
            "etrees","gboost","randomforest"]
    _o, _xparams, _yparam = db.load_data_for_deterministic_reg()
    _oh = _o[_o[_yparam[0]] >= 5.5]
    _ol = _o[_o[_yparam[0]] <= 5.5]
    Xh = _oh.as_matrix(_xparams)
    Xl = _ol.as_matrix(_xparams)
    yh = np.array(_oh[_yparam[0]].tolist()).reshape((len(_oh),1))
    yl = np.array(_ol[_yparam[0]].tolist()).reshape((len(_ol),1))
    f_clf = "out/rf.pkl"
    clf = util.get_best_determinsistic_classifier(f_clf)
    print("--> Classifier loaded..")
    print(Xh.shape,yh.shape,Xl.shape,yl.shape)
    Xh_train, Xh_test, yh_train, yh_test = train_test_split(Xh, yh, test_size=1.0/3.0, random_state=42)
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl, yl, test_size=1.0/3.0, random_state=42)
    prh = clf.predict_proba(Xh_test)[:,0]
    prl = clf.predict_proba(Xl_test)[:,0]
    y_obs = []
    y_obs.extend(yh_test[:,0].tolist())
    y_obs.extend(yl_test[:,0].tolist())
    for r in regs:
        yl_pred = []
        yh_pred = []
        y_pred = []
        print("Model:"+r)
        regO = util.get_regressor(r, 0)
        regL = util.get_regressor(r, 0)
        RL = regL[0]
        RL.fit(Xl_train,yl_train)
        RO = regO[0]
        RO.fit(Xh_train,yh_train)
        model = r
        for I,p in enumerate(prh):
            if p < .7: yh_pred.append(RL.predict(Xh_test[I,:].reshape((1,10))).ravel()[0])
            else: yh_pred.append(RO.predict(Xh_test[I,:].reshape((1,10))).ravel()[0])
            pass
        for I,p in enumerate(prl):
            if p < .7: yl_pred.append(RL.predict(Xl_test[I,:].reshape((1,10))).ravel()[0])
            else: yl_pred.append(RO.predict(Xl_test[I,:].reshape((1,10))).ravel()[0])
            pass
        y_pred.extend(yh_pred)
        y_pred.extend(yl_pred)
        _eval_details = util.run_validation(y_pred,y_obs,"[1995-2016]",model)
        print _eval_details
        #break
        pass
    return

def run_deterministic_clf_reg_model(args):
    if len(args) == 0: print "python jobutil.py 2 <reg model> <trw> <year>(1995-2016)"
    else:
        model = args[0]
        trw = int(args[1])
        if model == "dtree" or model == "etree" or model == "knn" or model == "ada":
            if args[2] == "all":
                for y in range(1995,2017): M.run_model_process_based_on_deterministic_algoritms(y, model, trw=trw)
                pass
            else:
                y = int(args[2])
                M.run_model_process_based_on_deterministic_algoritms(y, model, trw=trw)
            pass
        else:
            if args[2] == "all":
                for y in range(1995,2017): M.run_model_process_based_on_deterministic_algoritms(y, model, trw=trw)#M.run_model_based_on_deterministic_algoritms(y, model, trw=trw)
                pass
            else:
                y = int(args[2])
                #M.run_model_based_on_deterministic_algoritms(y, model, trw=trw)
                M.run_model_process_based_on_deterministic_algoritms(y, model, trw=trw)
            pass
        pass
    return

def run_gp_clf_reg_model(args):
    if len(args) == 0: print "python jobutil.py 3 GPR <ktype>(RBF/RQ/Matern) <trw> <year>(1995-2016)"
    else:
        model = args[0]
        ktype = args[1]
        trw = int(args[2])
        if args[3] == "all":
            for y in range(1995,2017): M.run_model_based_on_gp(y, kt=ktype, trw = trw)
            pass
        else:
            y = int(args[3])
            M.run_model_based_on_gp(y,  kt=ktype, trw = trw)
        pass
    return

def run_lstm_clf_reg_model(args):
    if len(args) == 0: print "python jobutil.py 5 LSTM <trw> <year>(1995-2016)"
    else:
        model = args[0]
        trw = int(args[1])
        if args[2] == "all":
            for y in range(1995,2017): M.run_model_based_on_lstm(y, model="LSTM", trw = trw)
            pass
        else:
            y = int(args[2])
            M.run_model_based_on_lstm(y, model="LSTM", trw = trw)
        pass
    return

def run_model_stats(args):
    if len(args) == 0: print "python jobutil.py 4 <reg model> <trw>"
    else:
        model = args[0]
        trw = int(args[1])
        util.get_stats(model, trw)
    return


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0: print "Invalid call sequence!! python jobutil.py {1/2/3...}"
    else:
        ctx = int(args[0])
        if ctx == 1: build_all_classifiers(args[1])
        if ctx == 2: run_deterministic_clf_reg_model(args[1:])
        if ctx == 3: run_gp_clf_reg_model(args[1:])
        if ctx == 4: run_model_stats(args[1:])
        if ctx == 5: run_lstm_clf_reg_model(args[1:])
        if ctx == 9: build_all_regressor()
        pass
    pass
    
