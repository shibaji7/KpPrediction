##
#
##

import os
import matplotlib
matplotlib.use("Agg")
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
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
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor


from sklearn.externals import joblib
from spacepy import plot as splot
from imblearn.under_sampling import RandomUnderSampler
import verify
from verify import Contingency2x2

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

def plot_deterministic_roc_curves(roc_eval_details):
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
    fig.savefig("out/deterministinc_forecast_models_roc_curves.png",bbox_inches="tight")
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
    REGs["elasticnet"] = (ElasticNet(alpha=.1,tol=1e-6), name, trw)
    REGs["bayesianridge"] = (BayesianRidge(n_iter=300, tol=1e-5, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept=True), name, trw)
    
    # decission trees
    REGs["dtree"] = (DecisionTreeRegressor(random_state=0), name, trw)
    REGs["etree"] = (ExtraTreeRegressor(random_state=0), name, trw)
    
    # NN regressor
    REGs["knn"] = (KNeighborsRegressor(n_neighbors=25,weights="distance"), name, trw)
    REGs["rnn"] = (RadiusNeighborsRegressor(radius=20.0), name, trw)
    
    # ensamble models
    REGs["ada"] = (AdaBoostRegressor(), name, trw)
    REGs["bagging"] = (BaggingRegressor(n_estimators=50, max_features=3), name, trw)
    REGs["etrees"] = (ExtraTreesRegressor(n_estimators=50), name, trw)
    REGs["gboost"] = (GradientBoostingRegressor(max_depth=5,random_state=0), name, trw)
    REGs["randomforest"] = (RandomForestRegressor(n_estimators=100), name, trw)
    return REGs[name]

def __run_validation(pred,obs,year,model):
    pred,obs = np.array(pred),np.array(obs)
    _eval_details = {}
    try: _eval_details["bias"] = verify.bias(pred,obs)
    except: _eval_details["bias"] = np.NaN
    try: _eval_details["meanPercentageError"] = verify.meanPercentageError(pred,obs)
    except: _eval_details["meanPercentageError"] = np.NaN
    try: _eval_details["medianLogAccuracy"] = verify.medianLogAccuracy(pred,obs)
    except: _eval_details["medianLogAccuracy"] = np.NaN
    try:_eval_details["symmetricSignedBias"] = verify.symmetricSignedBias(pred,obs)
    except: _eval_details["symmetricSignedBias"] = np.NaN
    try: _eval_details["meanSquaredError"] = verify.meanSquaredError(pred,obs)
    except: _eval_details["meanSquaredError"] = np.NaN
    try: _eval_details["RMSE"] = verify.RMSE(pred,obs)
    except: _eval_details["RMSE"] = np.NaN
    try: _eval_details["meanAbsError"] = verify.meanAbsError(pred,obs)
    except: _eval_details["meanAbsError"] = np.NaN
    try: _eval_details["medAbsError"] = verify.medAbsError(pred,obs)
    except: _eval_details["medAbsError"] = np.NaN
    
    try: _eval_details["nRMSE"] = verify.nRMSE(pred,obs)
    except: _eval_details["nRMSE"] = np.NaN
    try: _eval_details["forecastError"] = np.mean(verify.forecastError(pred,obs))
    except: _eval_details["forecastError"] = np.NaN
    try: _eval_details["logAccuracy"] = np.mean(verify.logAccuracy(pred,obs))
    except: _eval_details["logAccuracy"] = np.NaN
    
    try: _eval_details["medSymAccuracy"] = verify.medSymAccuracy(pred,obs)
    except: _eval_details["medSymAccuracy"] = np.NaN
    try: _eval_details["meanAPE"] = verify.meanAPE(pred,obs)
    except: _eval_details["meanAPE"] = np.NaN
    try: _eval_details["medAbsDev"] = verify.medAbsDev(pred)
    except: _eval_details["medAbsDev"] = np.NaN
    try: _eval_details["rSD"] = verify.rSD(pred)
    except: _eval_details["rSD"] = np.NaN
    try: _eval_details["rCV"] = verify.rCV(pred)
    except: _eval_details["rCV"] = np.NaN
    _eval_details["year"] = year
    _eval_details["model"] = model
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
