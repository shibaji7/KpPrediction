import datetime as dt
import util
import models
import database as db


def caseI():
    """
    Thread testing
    """
    f_clf = "out/rf.pkl"
    reg = util.get_regressor("dummy", 27)
    clf = util.get_best_determinsistic_classifier(f_clf)
    _o, _xparams, _yparam = db.load_data_for_deterministic_reg()
    data = (_o, _xparams, _yparam)
    dn = dt.datetime(1995,6,24,15)
    th = models.DeterministicModelPerDataPoint(1995,reg,clf,dn,data)
    th.start()
    return

caseI()
