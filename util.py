##
#
##

import os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['agg.path.chunksize'] = 10000
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import math
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

np.random.seed(7)
import database as db

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic as RQ

from sklearn.externals import joblib
from spacepy import plot as splot
from imblearn.under_sampling import RandomUnderSampler
import verify
from verify import Contingency2x2

from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding,Dropout

def nan_helper(y):
    nans = np.isnan(y)
    x = lambda z: z.nonzero()[0]
    f = interp1d(x(~nans), y[~nans], kind="cubic")
    y[nans] = f(x(nans))
    return y

def smooth(x,window_len=51,window="hanning"):
    if x.ndim != 1: raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len: raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[d/2:-d/2]
    return y

def get_classifiers():
    # basic classifires
    dc = DummyClassifier(random_state=0)
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1)
    C0 = {"name":"Basic","methods":[(dc, "Dummy"), (lr, "Logit"),(gnb, "Naive Bayes"), (svc, "SVC")]}
    
    # decission trees
    dec_tree = DecisionTreeClassifier(random_state=0)
    etc_tree = ExtraTreeClassifier(random_state=0)
    C1 = {"name":"Decision Tree","methods":[(dec_tree, "Decision Tree"),(etc_tree, "Extra Tree")]}
    
    # NN classifirer
    knn = KNeighborsClassifier(n_neighbors=25,weights="distance")
    rnn = RadiusNeighborsClassifier(radius=20.0,outlier_label=1)
    nc = NearestCentroid()
    C2 = {"name":"Nearest Neighbors","methods":[(knn, "KNN"),(rnn, "Radius NN"),(nc, "Nearest Centroid")]}
    
    
    # ensamble models
    ada = AdaBoostClassifier()
    bg = BaggingClassifier(n_estimators=50, max_features=3)
    etsc = ExtraTreesClassifier(n_estimators=50,criterion="entropy")
    gb = GradientBoostingClassifier(max_depth=5,random_state=0)
    rfc = RandomForestClassifier(n_estimators=100)
    C3 = {"name":"Ensamble","methods":[(ada, "Ada Boost"),(bg,"Bagging"),(etsc, "Extra Trees"),
        (gb, "Gradient Boosting"), (rfc, "Random Forest")]}

    # discriminant analysis & GPC
    lda = LinearDiscriminantAnalysis()
    qda = QuadraticDiscriminantAnalysis()
    C4 = {"name":"Discriminant Analysis","methods":[(lda, "LDA"),(qda, "QDA")]}
    
    # neural net
    nn = MLPClassifier(alpha=0.1,tol=1e-8)
    C5 = {"name":"Complex Architecture","methods":[(nn, "Neural Network")]}
    
    CLF = [C0,C1,C2,C3,C4,C5]
    return CLF

def get_roc_details(clf, X_test, y_test):
    if hasattr(clf, "predict_proba"): y_score = clf.predict_proba(X_test)[:, 1]
    else:
        prob_pos = clf.decision_function(X_test)
        y_score = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        pass
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    return y_score, fpr, tpr, roc_auc

def plot_deterministic_roc_curves(roc_eval_details, tag):
    fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(15,10))
    fig.subplots_adjust(wspace=0.5,hspace=0.5)
    splot.style("spacepy")
    lw = 2
    I = 0
    for gname in roc_eval_details.keys():
        i,j = int(I/3), int(np.mod(I,3))
        ax = axes[i,j]
        clf_type = roc_eval_details[gname]
        for name in clf_type.keys():
            roc = roc_eval_details[gname][name]
            ax.plot(roc["fpr"], roc["tpr"], color=roc["c"], lw = lw, label="ROC curve (%s:area = %0.2f)" % (name,roc["roc_auc"]))
            pass
        ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_title(gname)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right",prop={"size": 7})
        I = I + 1
        pass
    fig.savefig("out/deterministinc_forecast_models_roc_curves_%s.png"%tag,bbox_inches="tight")
    return

def validate_model_matrices(clf, X_test, y_true):
    y_pred = clf.predict(X_test)
    CM = confusion_matrix(y_true, y_pred)
    C2x2 = Contingency2x2(CM.T)
    return C2x2


def get_regressor(name, trw=27):
    REGs = {}
    # basic regressor            
    REGs["dummy"] = (DummyRegressor(strategy="median"), name, trw)
    REGs["regression"] = (LinearRegression(), name, trw)
    REGs["elasticnet"] = (ElasticNet(alpha=.5,tol=1e-2), name, trw)
    REGs["bayesianridge"] = (BayesianRidge(n_iter=300, tol=1e-5, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept=True), name, trw)
    
    # decission trees
    REGs["dtree"] = (DecisionTreeRegressor(random_state=0,max_depth=5), name, trw)
    REGs["etree"] = (ExtraTreeRegressor(random_state=0,max_depth=5), name, trw)
    
    # NN regressor
    REGs["knn"] = (KNeighborsRegressor(n_neighbors=25,weights="distance"), name, trw)
    
    # ensamble models
    REGs["ada"] = (AdaBoostRegressor(), name, trw)
    REGs["bagging"] = (BaggingRegressor(n_estimators=50, max_features=3), name, trw)
    REGs["etrees"] = (ExtraTreesRegressor(n_estimators=50), name, trw)
    REGs["gboost"] = (GradientBoostingRegressor(max_depth=5,random_state=0), name, trw)
    REGs["randomforest"] = (RandomForestRegressor(n_estimators=100), name, trw)
    return REGs[name]

def get_hyp_param(kernel_type):
    hyp = {}
    if kernel_type == "RBF": hyp["l"] = 1.0
    if kernel_type == "RQ": 
        hyp["l"] = 1.0
        hyp["a"] = 0.1
    if kernel_type == "Matern": hyp["l"] = 1.0
    return hyp

def get_gpr(kernel_type, hyp, nrst = 10, trw=27):
    if kernel_type == "RBF": kernel = RBF(length_scale=hyp["l"],length_scale_bounds=(1e-02, 1e2))
    if kernel_type == "RQ": kernel = RQ(length_scale=hyp["l"],alpha=hyp["a"],length_scale_bounds=(1e-02, 1e2),alpha_bounds=(1e-2, 1e2))
    if kernel_type == "Matern": kernel = Matern(length_scale=hyp["l"],length_scale_bounds=(1e-02, 1e2), nu=1.4)
    gpr = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = nrst)
    return (gpr, "GPR", trw)

def get_lstm(ishape,look_back=1, trw = 27):
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, ishape)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return (model, "LSTM", trw)

def get_lstm_classifier(ishape):
    model = Sequential()
    model.add(Embedding(input_dim = 188, output_dim = 50, input_length = ishape))
    model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    return model

def run_validation(pred,obs,year,model):
    pred,obs = np.array(pred),np.array(obs)
    _eval_details = {}
    _eval_details["range"] = "N"
    if max(pred) > 9. or min(pred) < 0.: _eval_details["range"] = "Y"
    try: _eval_details["bias"] = np.round(verify.bias(pred,obs),2)
    except: _eval_details["bias"] = np.NaN
    try: _eval_details["meanPercentageError"] = np.round(verify.meanPercentageError(pred,obs),2)
    except: _eval_details["meanPercentageError"] = np.NaN
    try: _eval_details["medianLogAccuracy"] = np.round(verify.medianLogAccuracy(pred,obs),3)
    except: _eval_details["medianLogAccuracy"] = np.NaN
    try:_eval_details["symmetricSignedBias"] = np.round(verify.symmetricSignedBias(pred,obs),3)
    except: _eval_details["symmetricSignedBias"] = np.NaN
    try: _eval_details["meanSquaredError"] = np.round(verify.meanSquaredError(pred,obs),2)
    except: _eval_details["meanSquaredError"] = np.NaN
    try: _eval_details["RMSE"] = np.round(verify.RMSE(pred,obs),2)
    except: _eval_details["RMSE"] = np.NaN
    try: _eval_details["meanAbsError"] = np.round(verify.meanAbsError(pred,obs),2)
    except: _eval_details["meanAbsError"] = np.NaN
    try: _eval_details["medAbsError"] = np.round(verify.medAbsError(pred,obs),2)
    except: _eval_details["medAbsError"] = np.NaN
    
    try: _eval_details["nRMSE"] = np.round(verify.nRMSE(pred,obs),2)
    except: _eval_details["nRMSE"] = np.NaN
    try: _eval_details["forecastError"] = np.round(np.mean(verify.forecastError(pred,obs)),2)
    except: _eval_details["forecastError"] = np.NaN
    try: _eval_details["logAccuracy"] = np.round(np.mean(verify.logAccuracy(pred,obs)),2)
    except: _eval_details["logAccuracy"] = np.NaN
    
    try: _eval_details["medSymAccuracy"] = np.round(verify.medSymAccuracy(pred,obs),2)
    except: _eval_details["medSymAccuracy"] = np.NaN
    try: _eval_details["meanAPE"] = np.round(verify.meanAPE(pred,obs),2)
    except: _eval_details["meanAPE"] = np.NaN
    try: _eval_details["medAbsDev"] = np.round(verify.medAbsDev(pred),2)
    except: _eval_details["medAbsDev"] = np.NaN
    try: _eval_details["rSD"] = np.round(verify.rSD(pred),2)
    except: _eval_details["rSD"] = np.NaN
    try: _eval_details["rCV"] = np.round(verify.rCV(pred),2)
    except: _eval_details["rCV"] = np.NaN
    _eval_details["year"] = year
    _eval_details["model"] = model
    r,_ =  pearsonr(pred,obs)
    _eval_details["r"] = r
    return _eval_details

def get_best_determinsistic_classifier(f_clf):
    if not os.path.exists(f_clf):
        # Dataset
        _xparams,X,y = db.load_data_for_deterministic_bin_clf()
        rus = RandomUnderSampler(return_indices=True)
        X_resampled, y_resampled, idx_resampled = rus.fit_sample(X, y)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_resampled,y_resampled)
        joblib.dump(clf, f_clf)
    else:
        clf = joblib.load(f_clf)
    return clf

def get_stats(model, trw):
    fname = "out/det.%s.pred.%d.csv"%(model,trw)
    fname = "out/det.%s.pred.%d.g.csv"%(model,trw)
    _o = pd.read_csv(fname)
    _o = _o[(_o.prob_clsf != -1.) & (_o.y_pred != -1.) & (_o.y_pred >= 0) & (_o.y_pred <= 9.)]
    y_pred = _o.y_pred.tolist()
    y_obs = _o.y_obs.tolist()
    _eval_details =  run_validation(y_pred,y_obs,"[1995-2016]",model)
    print _eval_details 
    splot.style("spacepy")
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
    ax.plot(y_pred,y_obs,"k.")
    print("Updated")
    strx = "RMSE=%.2f\nr=%.2f"%(_eval_details["RMSE"],_eval_details["r"])
    ax.text(0.2,0.95,strx,horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
    ax.set_xlabel(r"$K_{P_{pred}}$")
    ax.set_xlim(0,9)
    ax.set_ylim(0,9)
    ax.set_ylabel(r"$K_{P_{obs}}$")
    fig.savefig("out/stat/det.%s.pred.%d.png"%(model,trw))
    return


def run_for_TSS(model, trw):
    fdummy = "out/det.dummy.pred.%d.csv"%(trw)
    fname = "out/det.%s.pred.%d.csv"%(model,trw)
    _od = pd.read_csv(fdummy)
    _o = pd.read_csv(fname)
    _od = _od[(_od.prob_clsf != -1.) & (_od.y_pred != -1.) & (_od.y_pred >= 0) & (_od.y_pred <= 9.)]
    _o = _o[(_o.prob_clsf != -1.) & (_o.y_pred != -1.) & (_o.y_pred >= 0) & (_o.y_pred <= 9.)]
    _od.dn = pd.to_datetime(_od.dn)
    _o.dn = pd.to_datetime(_o.dn)

    stime = dt.datetime(1995,2,1)
    etime = dt.datetime(2016,9,20)
    d = stime
    skill = []
    t = []
    while(d < etime):
        try:
            t.append(d)
            dn = d + dt.timedelta(days=27)
            dum = _od[(_od.dn >= d) & (_od.dn < dn)]
            mod = _o[(_o.dn >= d) & (_o.dn < dn)]
            rmse_dum = verify.RMSE(dum.y_pred,dum.y_obs)
            rmse = verify.RMSE(mod.y_pred,mod.y_obs)
            print(d,rmse,rmse_dum,verify.skill(rmse, rmse_dum))
            skill.append(verify.skill(rmse, rmse_dum))
            d = d + dt.timedelta(days=1)
        except: pass
        pass
    skill = np.array(skill)
    skill = nan_helper(skill)
    fmt = matplotlib.dates.DateFormatter("%d %b\n%Y")
    splot.style("spacepy")
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6))
    ax.xaxis.set_major_formatter(fmt)
    ax.plot(t,skill,"k.",label="")
    ax.plot(t,smooth(np.array(skill),101),"r-.")
    #strx = "RMSE:%.2f\nr:%.2f"%(_eval_details["RMSE"],_eval_details["r"])
    #ax.text(0.2,0.8,strx,horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
    ax.set_ylabel(r"$TSS(\%)$")
    ax.set_xlabel(r"$Time$")
    ax.set_xlim(dt.datetime(1995,1,1), dt.datetime(2017,1,1))
    #ax.set_ylim(0,100)
    fig.savefig("out/stat/det.%s.tss.%d.png"%(model,trw)) 
