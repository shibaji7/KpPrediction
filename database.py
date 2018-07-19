##
# Import library
##

import os
import pandas as pd
import numpy as np
import scipy.io as io
from scipy.stats import boxcox
import datetime as dt
import time
import scipy.stats as stats

##
# Get Kp value
##
def read_kp(f_csv = "data_store/kp.csv"):
    _k = pd.read_csv(f_csv)
    _k.dates = pd.to_datetime(_k.dates)
    return _k

##
# Filter Omni data
##
def filterX(_o):
    _o = _o[(_o.Bx_m!=9999.99) & (_o.By_m!=9999.99) & (_o.Bz_m!=9999.99) & (_o.V_m!=99999.9) & (_o.Vx_m!=99999.9)
            & (_o.Vy_m!=99999.9) & (_o.Vz_m!=99999.9) & (_o.PR_d_m!=999.99) & (_o["T_m"]!=9999999.0) & (_o.P_dyn_m!=99.99)
            & (_o.E_m!=999.99) & (_o.beta_m!=999.99) & (_o.Ma_m!=999.9)]
    return _o

##
# read GOES
##
def read_goes(f_csv = "data_store/goes.csv"):
    _g = pd.read_csv(f_csv)
    _g.times = pd.to_datetime(_g.times)
    return _g

##
# Get Omni values (3h Average Data)
##
def read_omni_data(f_csv = "data_store/omni_3h.csv"):
    _o = pd.read_csv(f_csv)
    _o.sdates = pd.to_datetime(_o.sdates)
    _o.edates = pd.to_datetime(_o.edates)
    _o = filterX(_o)
    return _o

##
# Transform Kp
##
def do_transform_Kp2lin(Kp):
    _levels = ["0","0+","1-","1","1+","2-","2","2+","3-","3","3+","4-","4","4+",
            "5-","5","5+","6-","6","6+","7-","7","7+","8-","8","8+","9-","9"]
    _n = 0.33
    _lin_values = [0,0+_n,1-_n,1,1+_n,2-_n,2,2+_n,3-_n,3,3+_n,4-_n,4,4+_n,
            5-_n,5,5+_n,6-_n,6,6+_n,7-_n,7,7+_n,8-_n,8,8+_n,9-_n,9]
    _dict = dict(zip(_levels,_lin_values))
    _Kp_lin = []
    for _k in Kp: _Kp_lin.append(_dict[_k])
    _Kp_lin = np.array(_Kp_lin)
    return _Kp_lin

##
# Transformed variable
##
def transform_variables(_df):
    B_x = np.array(_df["Bx_m"]).T
    B_T = np.sqrt(np.array(_df["By_m"])**2+np.array(_df["Bz_m"])**2).T
    theta_c = np.arctan(np.array(_df["Bz_m"])/np.array(_df["By_m"])).T
    #theta_c[np.isnan(theta_c)] = 0.
    sinetheta_c2 = np.sin(theta_c/2)
    V = np.array(_df["V_m"]).T
    n = np.array(_df["PR_d_m"]).T
    T = np.array(_df["T_m"]).T
    P_dyn = np.array(_df["P_dyn_m"]).T
    beta = np.array(_df["beta_m"]).T
    M_A = np.array(_df["Ma_m"]).T
    Kp = np.array(_df["Kp"]).T
    Kplt = np.array(_df["_kp_lt"]).T
    Kpd = np.array(_df["_dkp"]).T
    Kpdlt = np.array(_df["_dkp_lt"]).T
    sdates = _df["sdates"]
    fcdates = _df["delay_time"]
    columns = ["B_x","B_T","theta_c","sin_tc","V","n","T",
            "P_dyn","beta","M_A","Date_WS","K_P","K_P_LT",
            "Date_FC","K_P_delay","K_P_LT_delay"]
    _o = pd.DataFrame(np.array([B_x,B_T,theta_c,sinetheta_c2,V,n,T,P_dyn,beta,M_A,sdates,
        Kp,Kplt,fcdates,Kpd,Kpdlt]).T,columns=columns)
    return _o

##
# Build X,y data for deterministic classifier model
##
def load_data_for_deterministic_bin_clf(th=4.5, mI=1):
    params = ["Bx_m","By_m","Bz_m","V_m","Vx_m","Vy_m","Vz_m","PR_d_m","T_m","P_dyn_m","E_m","beta_m",
            "Ma_m","sdates","Kp","_kp_lt","delay_time","_dkp","_dkp_lt"]
    headers = ["B_x","B_T","sin_tc","V","n","T",
            "P_dyn","beta","M_A","Date_WS","K_P","K_P_LT",
            "Date_FC","K_P_delay","K_P_LT_delay"]
    delay = 3*mI
    _o = read_omni_data()
    _k = read_kp()
    _dkp = []
    _kp = []
    delay_time = []
    for I,rec in _o.iterrows():
        now = rec["sdates"]
        FC_time = now + dt.timedelta(hours=delay)
        delay_time.append(FC_time)
        future_kp = _k[_k.dates == FC_time]
        now_kp = _k[_k.dates == now]
        if len(future_kp) == 0: _dkp.append(_dkp[-1])
        else: _dkp.append(future_kp.Kp.tolist()[0])
        if len(now_kp) == 0: _kp.append(_kp[-1])
        else: _kp.append(now_kp.Kp.tolist()[0])
        pass
    _dkp = np.array(_dkp)
    _kp = np.array(_kp)
    _o["_dkp"] = _dkp
    _o["Kp"] = _kp
    _o["delay_time"] = delay_time
    dkp_tx = do_transform_Kp2lin(_dkp)
    _o["_dkp_lt"] = dkp_tx
    _o["_kp_lt"] = do_transform_Kp2lin(_o.Kp)
    _o = transform_variables(_o) 
    stormL = np.zeros(len(_dkp))
    stormL[dkp_tx > th] = 1.
    _o["stormL"] = stormL
    _xparams = ["B_x","B_T","sin_tc","V","n","T",
            "P_dyn","beta","M_A","K_P_LT"]
    _yparam = ["stormL"]
    X = _o.as_matrix(_xparams)
    y = _o.as_matrix(_yparam)
    return _xparams, X, y

##
# Build dataframe data for deterministic regressor model
##
def load_data_for_deterministic_reg(mI=1):
    params = ["Bx_m","By_m","Bz_m","V_m","Vx_m","Vy_m","Vz_m","PR_d_m","T_m","P_dyn_m","E_m","beta_m",
            "Ma_m","sdates","Kp","_kp_lt","delay_time","_dkp","_dkp_lt"]
    headers = ["B_x","B_T","sin_tc","V","n","T",
            "P_dyn","beta","M_A","Date_WS","K_P","K_P_LT",
            "Date_FC","K_P_delay","K_P_LT_delay"]
    delay = 3*mI
    fname = "out/omni_3h_%d.csv"%delay
    if not os.path.exists(fname):
        _o = read_omni_data()
        _k = read_kp()
        _dkp = []
        _kp = []
        delay_time = []
        for I,rec in _o.iterrows():
            now = rec["sdates"]
            FC_time = now + dt.timedelta(hours=delay)
            delay_time.append(FC_time)
            future_kp = _k[_k.dates == FC_time]
            now_kp = _k[_k.dates == now]
            if len(future_kp) == 0: _dkp.append(_dkp[-1])
            else: _dkp.append(future_kp.Kp.tolist()[0])
            if len(now_kp) == 0: _kp.append(_kp[-1])
            else: _kp.append(now_kp.Kp.tolist()[0])
            pass
        _dkp = np.array(_dkp)
        _kp = np.array(_kp)
        _o["_dkp"] = _dkp
        _o["Kp"] = _kp
        _o["delay_time"] = delay_time
        dkp_tx = do_transform_Kp2lin(_dkp)
        _o["_dkp_lt"] = dkp_tx
        _o["_kp_lt"] = do_transform_Kp2lin(_o.Kp)
        _o = transform_variables(_o)
        _o.to_csv(fname, index=False, header=True)
        pass
    else:
        _o = pd.read_csv(fname)
        _o.Date_FC = pd.to_datetime(_o.Date_FC)
        _o.Date_WS = pd.to_datetime(_o.Date_WS)
        pass
    _xparams = ["B_x","B_T","sin_tc","V","n","T",
            "P_dyn","beta","M_A","K_P_LT"]
    _yparam = ["K_P_LT_delay"]
    return _o, _xparams, _yparam
##
# Build X,y (goes) data for deterministic classifier model
##
def load_data_with_goes_for_deterministic_bin_clf(th=4.5, mI=1):
    params = ["Bx_m","By_m","Bz_m","V_m","Vx_m","Vy_m","Vz_m","PR_d_m","T_m","P_dyn_m","E_m","beta_m",
            "Ma_m","sdates","Kp","_kp_lt","delay_time","_dkp","_dkp_lt"]
    headers = ["B_x","B_T","sin_tc","V","n","T",
            "P_dyn","beta","M_A","Date_WS","K_P","K_P_LT",
            "Date_FC","K_P_delay","K_P_LT_delay"]
    delay = 3*mI
    _g = read_goes()
    print(_g.head())
    _o = read_omni_data()
    #_o = _o[_o.sdates<dt.datetime(1996,1,1)]
    _k = read_kp()
    _dkp = []
    _kp = []
    delay_time = []
    _goes_am = []
    _goes_bm = []
    _goes_as = []
    _goes_bs = []
    for I,rec in _o.iterrows():
        now = rec["sdates"]
        FC_time = now + dt.timedelta(hours=delay)
        delay_time.append(FC_time)
        now_g = _g[_g.times == now]
        print now
        if len(now_g) == 0:
            _goes_am.append(_goes_am[-1])
            _goes_bm.append(_goes_bm[-1])
            _goes_as.append(_goes_as[-1])
            _goes_bs.append(_goes_bs[-1])
        else:
            _goes_am.append(now_g._a_max.tolist()[0]) 
            _goes_bm.append(now_g._b_max.tolist()[0])
            _goes_as.append(now_g._a_std.tolist()[0])
            _goes_bs.append(now_g._b_std.tolist()[0])
        future_kp = _k[_k.dates == FC_time]
        now_kp = _k[_k.dates == now]
        if len(future_kp) == 0: _dkp.append(_dkp[-1])
        else: _dkp.append(future_kp.Kp.tolist()[0])
        if len(now_kp) == 0: _kp.append(_kp[-1])
        else: _kp.append(now_kp.Kp.tolist()[0])
        pass
    _dkp = np.array(_dkp)
    _kp = np.array(_kp)
    _o["_dkp"] = _dkp
    _o["Kp"] = _kp
    _o["delay_time"] = delay_time
    dkp_tx = do_transform_Kp2lin(_dkp)
    _o["_dkp_lt"] = dkp_tx
    _o["_kp_lt"] = do_transform_Kp2lin(_o.Kp)
    _o = transform_variables(_o) 
    _o["_a_max"] = _goes_am
    _o["_b_max"] = _goes_bm
    _o["_a_std"] = _goes_as
    _o["_b_std"] = _goes_bs
    stormL = np.zeros(len(_dkp))
    stormL[dkp_tx > th] = 1.
    _o["stormL"] = stormL
    _xparams = ["B_x","B_T","sin_tc","V","n","T",
            "P_dyn","beta","M_A","K_P_LT","_a_max","_b_max","_a_std","_b_std"]
    _yparam = ["stormL"]
    X = _o.as_matrix(_xparams)
    y = _o.as_matrix(_yparam)
    return _xparams, X, y

##
# Build dataframe data (goes) for deterministic regressor model
##
def load_data_with_goes_for_deterministic_reg(mI=1):
    params = ["Bx_m","By_m","Bz_m","V_m","Vx_m","Vy_m","Vz_m","PR_d_m","T_m","P_dyn_m","E_m","beta_m",
            "Ma_m","sdates","Kp","_kp_lt","delay_time","_dkp","_dkp_lt"]
    headers = ["B_x","B_T","sin_tc","V","n","T",
            "P_dyn","beta","M_A","Date_WS","K_P","K_P_LT",
            "Date_FC","K_P_delay","K_P_LT_delay"]
    delay = 3*mI
    fname = "out/omni_goes_3h_%d.csv"%delay
    if not os.path.exists(fname):
        _g = read_goes()
        _o = read_omni_data()
        _k = read_kp()
        _dkp = []
        _kp = []
        delay_time = []
        _goes_am = []
        _goes_bm = []
        _goes_as = []
        _goes_bs = []
        for I,rec in _o.iterrows():
            now = rec["sdates"]
            print now
            FC_time = now + dt.timedelta(hours=delay)
            delay_time.append(FC_time)
            future_kp = _k[_k.dates == FC_time]
            now_kp = _k[_k.dates == now]
            now_g = _g[_g.times == now]
            if len(now_g) == 0:
                _goes_am.append(_goes_am[-1])
                _goes_bm.append(_goes_bm[-1])
                _goes_as.append(_goes_as[-1])
                _goes_bs.append(_goes_bs[-1])
            else:
                _goes_am.append(now_g._a_max.tolist()[0])
                _goes_bm.append(now_g._b_max.tolist()[0])
                _goes_as.append(now_g._a_std.tolist()[0])
                _goes_bs.append(now_g._b_std.tolist()[0])
            if len(future_kp) == 0: _dkp.append(_dkp[-1])
            else: _dkp.append(future_kp.Kp.tolist()[0])
            if len(now_kp) == 0: _kp.append(_kp[-1])
            else: _kp.append(now_kp.Kp.tolist()[0])
            pass
        _dkp = np.array(_dkp)
        _kp = np.array(_kp)
        _o["_dkp"] = _dkp
        _o["Kp"] = _kp
        _o["delay_time"] = delay_time
        dkp_tx = do_transform_Kp2lin(_dkp)
        _o["_dkp_lt"] = dkp_tx
        _o["_kp_lt"] = do_transform_Kp2lin(_o.Kp)
        _o = transform_variables(_o)
        _o["_a_max"] = _goes_am
        _o["_b_max"] = _goes_bm
        _o["_a_std"] = _goes_as
        _o["_b_std"] = _goes_bs
        _o.to_csv(fname, index=False, header=True)
        pass
    else:
        _o = pd.read_csv(fname)
        _o.Date_FC = pd.to_datetime(_o.Date_FC)
        _o.Date_WS = pd.to_datetime(_o.Date_WS)
        pass
    _xparams = ["B_x","B_T","sin_tc","V","n","T",
            "P_dyn","beta","M_A","K_P_LT","_a_max","_b_max","_a_std","_b_std"]
    _yparam = ["K_P_LT_delay"]
    return _o, _xparams, _yparam


##
# Build X,y (goes) data for deterministic classifier model
##
def load_data_with_goes_for_lstm_bin_clf(th=4.5, mI=1, isgoes = False):
    params = ["Bx_m","By_m","Bz_m","V_m","Vx_m","Vy_m","Vz_m","PR_d_m","T_m","P_dyn_m","E_m","beta_m",
            "Ma_m","sdates","Kp","_kp_lt","delay_time","_dkp","_dkp_lt"]
    headers = ["B_x","B_T","sin_tc","V","n","T",
            "P_dyn","beta","M_A","Date_WS","K_P","K_P_LT",
            "Date_FC","K_P_delay","K_P_LT_delay"]
    delay = 3*mI
    _g = read_goes()
    print(_g.head())
    _o = read_omni_data()
    #_o = _o[_o.sdates<dt.datetime(1996,1,1)]
    _k = read_kp()
    _dkp = []
    _kp = []
    delay_time = []
    _goes_am = []
    _goes_bm = []
    _goes_as = []
    _goes_bs = []
    for I,rec in _o.iterrows():
        now = rec["sdates"]
        FC_time = now + dt.timedelta(hours=delay)
        delay_time.append(FC_time)
        now_g = _g[_g.times == now]
        if len(now_g) == 0:
            _goes_am.append(_goes_am[-1])
            _goes_bm.append(_goes_bm[-1])
            _goes_as.append(_goes_as[-1])
            _goes_bs.append(_goes_bs[-1])
        else:
            _goes_am.append(now_g._a_max.tolist()[0]) 
            _goes_bm.append(now_g._b_max.tolist()[0])
            _goes_as.append(now_g._a_std.tolist()[0])
            _goes_bs.append(now_g._b_std.tolist()[0])
        future_kp = _k[_k.dates == FC_time]
        now_kp = _k[_k.dates == now]
        if len(future_kp) == 0: _dkp.append(_dkp[-1])
        else: _dkp.append(future_kp.Kp.tolist()[0])
        if len(now_kp) == 0: _kp.append(_kp[-1])
        else: _kp.append(now_kp.Kp.tolist()[0])
        pass
    _dkp = np.array(_dkp)
    _kp = np.array(_kp)
    _o["_dkp"] = _dkp
    _o["Kp"] = _kp
    _o["delay_time"] = delay_time
    dkp_tx = do_transform_Kp2lin(_dkp)
    _o["_dkp_lt"] = dkp_tx
    _o["_kp_lt"] = do_transform_Kp2lin(_o.Kp)
    _o = transform_variables(_o)
    if isgoes:
        _o["_a_max"] = _goes_am
        _o["_b_max"] = _goes_bm
        _o["_a_std"] = _goes_as
        _o["_b_std"] = _goes_bs
        pass
    stormL = np.zeros(len(_dkp))
    stormL[dkp_tx > th] = 1.
    _o["stormL"] = stormL
    if isgoes:
        _xparams = ["B_x","B_T","sin_tc","V","n","T",
                "P_dyn","beta","M_A","K_P_LT","_a_max","_b_max","_a_std","_b_std"]
    else:
        _xparams = ["B_x","B_T","sin_tc","V","n","T",
                "P_dyn","beta","M_A","K_P_LT"]
    _yparam = ["stormL"]
    X = _o.as_matrix(_xparams)
    y = _o.as_matrix(_yparam)
    return _xparams, X, y

