import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import  StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
import numpy as np
import pickle
seed=42
np.random.seed(seed)
path= 'labels/_'

def xgbfun(XTrain, YTrain,estimator_max_depth= [7],weights = [2, 3, 4, 5],max_delta_step= [2, 3, 4, 5, 6], save=""):

    parameters = {"max_depth": estimator_max_depth,  # default 6
                  #"learning_rate":[0.2,0.3,0.4],
                  "max_delta_step":max_delta_step,
                  "scale_pos_weight": weights
                  }
    # https: // xgboost.readthedocs.io / en / stable / tutorials / param_tuning.html
    sCV = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    xgboost = xgb.XGBClassifier(verbosity=0)
    clf_xgboost = GridSearchCV(xgboost, parameters, cv=sCV, scoring='roc_auc')

    clf_xgboost.fit(XTrain, YTrain)

    model = clf_xgboost.best_estimator_
    score= clf_xgboost.best_score_
    print("Estimated:", model.max_delta_step, model.scale_pos_weight,model.max_depth,score,XTrain.shape)


    if(save!=""):
        pickle.dump(model, open(path+"XGBModel.pickle.dat", "wb"))

    return model,score,model.max_delta_step, model.scale_pos_weight,model.max_depth
def RFfun(XTrain, YTrain):
    parameters = {"max_depth": [7],

                  "class_weight": [{0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}]
    }
    sCV = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    rf = RandomForestClassifier(random_state=seed)
    clf_rf = GridSearchCV(rf, parameters, cv=sCV, scoring='roc_auc')
    clf_rf.fit(XTrain,np.ravel(YTrain))
    model = clf_rf.best_estimator_
    score = clf_rf.best_score_
    print("Estimated:", model.max_samples, model.criterion, model.max_features,model.class_weight)
    return model,score,model.max_samples, model.criterion, model.max_features,model.class_weight

def SVMfun(XTrain, YTrain):
    parameters = {
        "class_weight": [{0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}]
    }
    sCV = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    svc = SVC(random_state=seed,probability=True) ##Remove probability to speed up the computation
    #svc = SVC(random_state=seed)
    clf_svc = GridSearchCV(svc, parameters, cv=sCV, scoring='roc_auc')
    clf_svc.fit(XTrain,np.ravel(YTrain))
    model = clf_svc.best_estimator_
    score = clf_svc.best_score_
    print("Estimated:", model.kernel, model.gamma, model.class_weight)
    return model,score,model.kernel, model.gamma, model.class_weight


