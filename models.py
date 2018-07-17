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
from filelock import FileLock
from sklearn.preprocessing import MinMaxScaler

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

def store_prediction_to_loc_file(fname,dn,y_obs,y_pred,pr_c,prT,model):
    with FileLock(fname):
        if not os.path.exists(fname):
            with open(fname, "a+") as f: f.write("dn,y_obs,y_pred,prob_clsf,probT,model\n")
        with open(fname, "a+") as f: f.write("%s,%.2f,%.2f,%.2f,%.2f,%s\n"%(dn.strftime("%Y-%m-%d %H:%M:%S"),y_obs,y_pred,pr_c,prT,model))
        pass
    return


class ModelPerDataPoint(threading.Thread):
    def __init__(self, y, reg_det, clf, dn, data, alt_win):
        threading.Thread.__init__(self)
        self.y = y
        self.reg = reg_det[0]
        self.clf = clf
        self.dn = dn
        self.data = data
        self.trw = reg_det[2]
        self.mI = 1
        self.model = reg_det[1]
        self.alt_win = alt_win
        self.fname = "out/det.%s.pred.%d.g.csv"%(self.model,self.trw)
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
                    self.data_windowing(self.trw*self.alt_win, True)
                    _o_train = self._o_train
                    print(self.dn,pr)
                    X_train = _o_train.as_matrix(_xparams)
                    y_train = np.array(_o_train[_yparam]).reshape(len(_o_train),1)
                    pass
                reg.fit(X_train, y_train)
                if len(reg.predict(X_test).shape) == 2: self.y_pred = reg.predict(X_test)[0,0]
                else: self.y_pred = reg.predict(X_test)[0]
            except: 
                print(self.dn)
                traceback.print_exc()
            pass
        else: pass
        print(self.y_obs,self.y_pred)
        if self.model == "": store_prediction_to_loc_file(self.fname,self.dn,self.y_obs,self.y_pred,self.pr,self.prt,self.model)
        else: store_prediction_to_file(self.fname,self.dn,self.y_obs,self.y_pred,self.pr,self.prt,self.model)
        return

def run_model_per_year(details):
    y = details[0]
    reg = details[1]
    clf = details[2]
    data = details[3]
    alt_win = details[4]
    N = 8*30*8
    _dates = [dt.datetime(y,2,1) + dt.timedelta(hours=i*3) for i in range(N)]
    print("-->Process for year:%d"%y)
    for dn in _dates: 
        th = ModelPerDataPoint(y,reg,clf,dn,data,alt_win)
        th.start()
        pass
    return


###
# GLM
###
def run_model_based_on_deterministic_algoritms(Y, model, trw=27):
    print("--> Loading data...")
    #_o, _xparams, _yparam = db.load_data_for_deterministic_reg()
    _o, _xparams, _yparam = db.load_data_with_goes_for_deterministic_reg()
    f_clf = "out/rf.pkl"
    clf = util.get_best_determinsistic_classifier(f_clf)
    reg = util.get_regressor(model, trw)
    years = range(Y,Y+1)
    regs = [reg] * len(years)
    clfs = [clf] * len(years)
    alt_wins = [10] * len(years)
    data_array = [(_o, _xparams, _yparam)] * len(years)
    _a = []
    for x,y,z,k,aw in zip(years, regs, clfs, data_array, alt_wins): _a.append((x,y,z,k,aw))
    year_pool = Pool(10)
    year_pool.map(run_model_per_year, _a)
    return

def run_process_model_per_date(details):
    y = details[0]
    reg = details[1]
    clf = details[2]
    dn = details[3]
    data = details[4]
    alt_win = details[5]
    th = ModelPerDataPoint(y,reg,clf,dn,data,alt_win)
    th.run()
    return

def run_model_process_based_on_deterministic_algoritms(Y, model, trw=27):
    _o, _xparams, _yparam = db.load_data_for_deterministic_reg()
    f_clf = "out/rf.pkl"
    clf = util.get_best_determinsistic_classifier(f_clf)
    reg = util.get_regressor(model, trw)
    N = 8*30*8
    _dates = [dt.datetime(Y,2,1) + dt.timedelta(hours=i*3) for i in range(N)]
    print("-->Process for year:%d"%Y)
    years = [Y] * len(_dates)
    regs = [reg] * len(_dates)
    clfs = [clf] * len(_dates)
    alt_wins = [10] * len(_dates)
    data_array = [(_o, _xparams, _yparam)] * len(_dates)
    _a = []
    for x,y,z,dn,k,aw in zip(years, regs, clfs, _dates, data_array, alt_wins): _a.append((x,y,z,dn,k,aw))
    date_pool = Pool(12)
    date_pool.map(run_process_model_per_date, _a)    
    return

###
# GP models
###
def run_gp_model_per_date(details):
    y = details[0]
    reg = details[1]
    clf = details[2]
    dn = details[3]
    data = details[4]
    alt_win = details[5]
    th = ModelPerDataPoint(y,reg,clf,dn,data,alt_win)
    th.run()
    return

def run_model_based_on_gp(Y, kt = "RQ", model="GPR", trw=27):
    print("--> Loading data...")
    _o, _xparams, _yparam = db.load_data_for_deterministic_reg()
    f_clf = "out/rf.pkl"
    clf = util.get_best_determinsistic_classifier(f_clf)
    hyp = util.get_hyp_param(kt)
    reg = util.get_gpr(kt, hyp, nrst = 10, trw=27)
    N = 8*30*8
    _dates = [dt.datetime(Y,2,1) + dt.timedelta(hours=i*3) for i in range(N)]
    print("-->Process for year:%d"%Y)
    years = [Y] * len(_dates)
    regs = [reg] * len(_dates)
    clfs = [clf] * len(_dates)
    alt_wins = [36] * len(_dates)
    data_array = [(_o, _xparams, _yparam)] * len(_dates)
    _a = []
    for x,y,z,dn,k,aw in zip(years, regs, clfs, _dates, data_array, alt_wins): _a.append((x,y,z,dn,k,aw))
    date_pool = Pool(12)
    date_pool.map(run_gp_model_per_date, _a)
    return


###
#  LSTM model
###
class LSTMPerDataPoint(object):
    def __init__(self, y, reg_det, clf, dn, data, alt_win, look_back):
        self.y = y
        self.reg = reg_det[0]
        self.clf = clf
        self.dn = dn
        self.data = data
        self.trw = reg_det[2]
        self.mI = 1
        self.model = reg_det[1]
        self.alt_win = alt_win
        self.fname = "out/det.%s.pred.%d.csv"%(self.model,self.trw)
        self.sclX = MinMaxScaler(feature_range=(0, 1))
        self.sclY = MinMaxScaler(feature_range=(0, 1))
        self.look_back = look_back
        return

    def data_windowing(self, trw=None, isLW = False):
        _o = self.data[0]
        _xparams = self.data[1]
        _yparam = self.data[2]
        if trw is None: trw = self.trw
        _tstart = self.dn - dt.timedelta(days=trw) # training window start inclusive
        _tend = self.dn # training window end inclusive
        _o_all = _o[(_o["Date_WS"] >= _tstart) & (_o["Date_WS"] <= _tend)]
        if isLW: _o_all = _o_all[_o_all[_yparam] >= 4.5]
        _o_test  = _o_all[_o_all["Date_WS"]==self.dn]
        T = False
        if len(_o_test) == 1:
            X,y = _o_all[_xparams].as_matrix(), _o_all[_yparam].as_matrix()
            Xm,ym = self.txXY(X,y)
            self.X_test, self.y_test = Xm[-1,:], ym[-1,0]
            self.X_train, self.y_train  = Xm[:-1,:], ym[:-1,0].reshape((len(ym)-1,1))
            self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[2]))
            self.y_obs = self.reY([[self.y_test]])[0,0]
            self.X_test_lstm = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))
            print(self.X_train.shape,self.y_train.shape, self.X_test.shape)
            T = True
            pass
        return T

    def create_dataset(self,X,y, look_back=1):
        dataX, dataY = [], []
        for i in range(look_back+1,len(X)):
            a = X[i-look_back:i, :]
            dataX.append(a)
            dataY.append(y[i].tolist())
            pass
        return np.array(dataX), np.array(dataY)
    

    def txXY(self, X, y):
        Xs = self.sclX.fit_transform(X)
        ys = self.sclY.fit_transform(y)
        Xm,ym = self.create_dataset(Xs,ys,self.look_back)
        return Xm,ym

    def reY(self, y):
        y = self.sclY.inverse_transform(y)
        return y

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
        self.y_obs = -1
        self.y_pred = -1
        self.pr = -1
        self.prt = prt
        if self.data_windowing():
            X_train,y_train = self.X_train,self.y_train
            X_test,y_test = self.X_test,self.y_test
            try:
                pr = clf.predict_proba(X_test)[0,0]
                self.pr = pr
                if pr > prt:
                    self.data_windowing(self.trw*self.alt_win, True)
                    X_train,y_train = self.X_train,self.y_train
                    pass
                reg.fit(X_train, y_train, batch_size = 2, epochs = 50, verbose = 0)
                if len(reg.predict(self.X_test_lstm).shape) == 2: self.y_pred = self.reY(reg.predict(self.X_test_lstm)[0,0])[0,0]
                else: self.y_pred = self.reY(reg.predict(self.X_test_lstm)[0])[0,0]
            except: 
                print(self.dn)
                traceback.print_exc()
            pass
        else: pass
        print(self.y_obs,self.y_pred)
        store_prediction_to_file(self.fname,self.dn,self.y_obs,self.y_pred,self.pr,self.prt,self.model)
        return


def run_lstm_model_per_date(details):
    y = details[0]
    reg_details = details[1]
    look_back = reg_details[1]
    reg = util.get_lstm(ishape=reg_details[0],look_back=reg_details[1],trw=reg_details[2])
    clf = details[2]
    dn = details[3]
    data = details[4]
    alt_win = details[5]
    th = LSTMPerDataPoint(y,reg,clf,dn,data,alt_win,look_back)
    th.run()
    return

def run_model_based_on_lstm(Y, model="LSTM", trw=27):
    print("--> Loading data...")
    _o, _xparams, _yparam = db.load_data_for_deterministic_reg()
    f_clf = "out/rf.pkl"
    clf = util.get_best_determinsistic_classifier(f_clf)
    reg = (10,1,trw)
    N = 8*30*8
    _dates = [dt.datetime(Y,2,1) + dt.timedelta(hours=i*3) for i in range(N)]
    print("-->Process for year:%d"%Y)
    years = [Y] * len(_dates)
    regs = [reg] * len(_dates)
    clfs = [clf] * len(_dates)
    alt_wins = [36] * len(_dates)
    data_array = [(_o, _xparams, _yparam)] * len(_dates)
    _a = []
    for x,y,z,dn,k,aw in zip(years, regs, clfs, _dates, data_array, alt_wins): _a.append((x,y,z,dn,k,aw))
    date_pool = Pool(10)
    date_pool.map(run_lstm_model_per_date, _a)
    return
