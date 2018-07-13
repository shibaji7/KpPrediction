##
#
##

import os
import pandas

import util

def run_model_based_on_deterministic_algoritms(model):
    fname = "out/func.rmse.csv"
    H = ["year","model","bias","meanPercentageError","medianLogAccuracy","symmetricSignedBias","meanSquaredError","RMSE","meanAbsError",\
            "medAbsError","nRMSE","forecastError","logAccuracy", "medSymAccuracy","meanAPE","medAbsDev","rSD","rCV"]
    if os.path.exists(fname): D = pd.read_csv(fname)
    else: D = pd.DataFrame()
    clf = util.get_best_classifier() 
    return
