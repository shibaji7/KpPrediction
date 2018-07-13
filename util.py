##
#
##

import os
import matplotlib
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
    return y_score, fpr, tpr, roc_au

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
