##
#
##

import os
import pandas as pd
import datetime as dt
import numpy as np
import multiprocessing
from multiprocessing import Pool
import threading
import traceback

import util
import database as db

global lock
lock = multiprocessing.Lock()

def store_prediction_to_file(fname,dn,y_obs,y_pred,pr_c,prT,model):
    global lock
    with lock:
        if not os.path.exists(fname):
            with open(fname, "a+") as f: f.write("dn,y_obs,y_pred,prob_clsf,probT,model\n")
        with open(fname, "a+") as f: f.write("%s,%.2f,%.2f,%.2f,%.2f,%s\n"%(dn.strftime("%Y-%m-%d %H:%M:%S"),y_obs,y_pred,pr_c,prT,model))
        pass
    return


class DeterministicModelPerDataPoint(threading.Thread):
    def __init__(self, y, reg_det, clf, dn, data):
        threading.Thread.__init__(self)
        self.y = y
        self.reg = reg_det[0]
        self.clf = clf
        self.dn = dn
        self.data = data
        self.trw = reg_det[2]
        self.mI = 1
        self.model = reg_det[1]
        self.fname = "out/det.%s.pred.csv"%self.model
        return

    def data_windowing(self, trw=None, isLW = False):
        _o = self.data[0]
        _xparams = self.data[1]
        _yparam = self.data[2]
        if trw is None: trw = self.trw
        _tstart = self.dn - dt.timedelta(days=trw) # training window start inclusive
        _tend = self.dn - dt.timedelta(hours=3) # training window end inclusive
        self._o_train = _o[(_o["Date_WS"] >= _tstart) & (_o["Date_WS"] <= _tend)]
        self._o_test = _o[(_o["Date_WS"] == self._pred_point_time)]
        if isLW: 
            _o_train = self._o_train[self._o_train[_yparam] >= 4.5]
            if  np.count_nonzero(_o_train.as_matrix(_xparams)) == 0: self._o_train = _o_train
            pass
        return

    def run(self):
        prt = 0.7
        print("-->Process for date:%s"%self.dn)
        _xparams = self.data[1]
        _yparam = self.data[2]
        mI = self.mI
        reg = self.reg
        clf = self.clf
        self._forecast_time = self.dn + dt.timedelta(hours = (mI*3))
        self._pred_point_time = self.dn # Time at which forecast is taking place
        self.data_windowing()
        _o_train = self._o_train
        _o_test = self._o_test
        X_test = _o_test.as_matrix(_xparams)
        self.y_obs = -1
        self.y_pred = -1
        self.pr = -1
        self.prt = prt
        if _o_test.shape[0] == 1:
            try:
                X_train = _o_train.as_matrix(_xparams)
                y_train = np.array(_o_train[_yparam]).reshape(len(_o_train),1)
                X_test = _o_test.as_matrix(_xparams)
                y_test = np.array(_o_test[_yparam]).reshape(len(_o_test),1)
                self.y_obs = y_test[0,0]
                pr = clf.predict_proba(X_test)[0,0]
                self.pr = pr
                if pr > prt:
                    self.data_windowing(self.trw*10, True)
                    _o_train = self._o_train
                    print(self.dn,pr)
                    X_train = _o_train.as_matrix(_xparams)
                    y_train = np.array(_o_train[_yparam]).reshape(len(_o_train),1)
                    pass
                reg.fit(X_train, y_train)
                if len(reg.predict(X_test).shape) == 2: self.y_pred = reg.predict(X_test)[0,0]
                else: self.y_pred = reg.predict(X_test)[0]
            except: traceback.print_exc()
            pass
        else: pass
        print(self.y_obs,self.y_pred)
        store_prediction_to_file(self.fname,self.dn,self.y_obs,self.y_pred,self.pr,self.prt,self.model)
        return

def run_detrmnstic_model_per_year(details):
    y = details[0]
    reg = details[1]
    clf = details[2]
    data = details[3]
    N = 8*30*8
    #N = 1
    _dates = [dt.datetime(y,2,1) + dt.timedelta(hours=i*3) for i in range(N)]
    print("-->Process for year:%d"%y)
    for dn in _dates: 
        th = DeterministicModelPerDataPoint(y,reg,clf,dn,data)
        th.start()
        pass
    return

def run_model_based_on_deterministic_algoritms(Y, model, trw=27):
    print("--> Loading data...")
    _o, _xparams, _yparam = db.load_data_for_deterministic_reg()
    f_clf = "out/rf.pkl"
    clf = util.get_best_determinsistic_classifier(f_clf)
    reg = util.get_regressor(model, trw)
    years = range(Y,Y+1)
    regs = [reg] * len(years)
    clfs = [clf] * len(years)
    data_array = [(_o, _xparams, _yparam)] * len(years)
    _a = []
    for x,y,z,k in zip(years, regs, clfs, data_array): _a.append((x,y,z,k))
    year_pool = Pool(10)
    year_pool.map(run_detrmnstic_model_per_year, _a)
    return

