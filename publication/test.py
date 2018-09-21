import datetime as dt
import util
import models
import database as db


def caseI():
    """
    Thread testing
    """
    f_clf = "out/rf.pkl"
    reg = util.get_regressor("dtree", 27)
    clf = util.get_best_determinsistic_classifier(f_clf)
    _o, _xparams, _yparam = db.load_data_for_deterministic_reg()
    data = (_o, _xparams, _yparam)
    dn = dt.datetime(1995,6,24,15)
    th = models.ModelPerDataPoint(1995,reg,clf,dn,data,10)
    th.start()
    return

def caseII():
    """
    Thread gp testing
    """
    _o, _xparams, _yparam = db.load_data_for_deterministic_reg()
    f_clf = "out/rf.pkl"
    kt = "RQ"
    clf = util.get_best_determinsistic_classifier(f_clf)
    hyp = util.get_hyp_param(kt)
    reg = util.get_gpr(kt, hyp, nrst = 10, trw=27)
    _o, _xparams, _yparam = db.load_data_for_deterministic_reg()
    data = (_o, _xparams, _yparam)
    dn = dt.datetime(1995,6,24,15)
    th = models.ModelPerDataPoint(1995,reg,clf,dn,data,0.25)
    th.start() 
    return

caseI()
