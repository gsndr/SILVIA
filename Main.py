#from mlxtend.evaluate import PredefinedHoldoutSplit
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from learning import xgbfun, RFfun, SVMfun
from loading import loaddataset, loaddataset_of_indexes, normalize, concatenatedataset
from predicting import predictimagewithdistillationGlobal, predictimage
import numpy as np
from datetime import datetime
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#from catboost import CatBoostClassifier
import  pandas as pd
seed =42
np.random.seed(seed)

def mutualInfoRank(data,independentList,label):
    #import pandas as pd
    from sklearn.feature_selection import mutual_info_classif
    #df = pd.DataFrame(data, columns=independentList)
    res = dict(zip(independentList,
                   mutual_info_classif(data, label,discrete_features=False, random_state=seed)
                   ))
    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    print(sorted_x)
    return sorted_x


def classification(indexes,train,test,fold,scaling=False,modelType=0,tytpe=0,Normalize=False,):
    Mean=[]
    Stdev=[]
    import time
    #modelType=0 -- XGB
    #modelType=1 -- RF
    #modelType=2 -- SVM
    XTrainOri,YTrain,MeanOri,StdevOri=loaddataset(train,type,Normalize) #load original data
    if(len(indexes)>0):
        XTrainIndexes,YTrainIndexes,MeanIndexes,StdevIndexes=loaddataset_of_indexes(train,indexes,Normalize)
        XTrain=concatenatedataset(XTrainOri,XTrainIndexes)
    else:
        XTrain=XTrainOri

    minmax = MinMaxScaler()
    if(scaling==True):
        XTrain=minmax.fit_transform(XTrain)

    print(XTrain.shape)
    start = time.time()

    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")

    if(modelType==0):
        model,score,max_delta_step, scale_pos_weight,estimator_max_dept=xgbfun(XTrain,YTrain)
        modelname="XGB"
    elif(modelType==1):
        model, score,max_samples, criterion, max_features,class_weight = RFfun(XTrain,YTrain)
        modelname = "RF"
    elif (modelType==2):
        model, score, kernel, gamma, class_weight = SVMfun(XTrain,YTrain)
        modelname = "SVM"

    print("Score",score)

    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")

    end = time.time()

    ##### Generate predictions for each test image and print the map
    import pickle
    nameReport=modelname+"classification"+str(fold)
    if(len(indexes)>0):
        nameReport=nameReport+"_Indexes"
    with open(nameReport+".txt", "w") as f:
        print("Time spent training the classification model", (end-start), file=f)
        print("Start Time =", start_time, file=f)
        print("End Time =", end_time, file=f)
        print("Start Time =", start_time)
        print("End Time =", end_time)
        print("Indexes",indexes,file=f)
        if(modelType==0):
            print("XGB:", max_delta_step, scale_pos_weight, estimator_max_dept)
            print("XGB:", max_delta_step, scale_pos_weight, estimator_max_dept, file=f)
        elif(modelType==1):
            print("RF:",max_samples, criterion, max_features,class_weight)
            print("RF:", max_samples, criterion, max_features, class_weight, file=f)
        elif(modelType==2):
            print("SVM:", kernel, gamma, class_weight )
            print("SVM:", kernel, gamma, class_weight, file=f)
        for R in np.arange(1,2,1):
            print("test confusion matrix with radius",R)

            if (len(indexes) > 0):
                name=modelname+"_Indexes"
            else:
                name=modelname
            predictimage(scaling,minmax,f,test, model, type,name,indexes, R,Normalize,Mean,Stdev)
            namepickle=name+"Model_R"+str(R)+"_fold"+str(fold)
            if (len(indexes) > 0):
                namepickle = namepickle + "_Indexes"
            pickle.dump(model, open(namepickle+".pickle.dat", "wb"))

            print("test confusion matrix with self training and radius",R)

            if (len(indexes) > 0):
                name = modelname + "_Indexes"
            else:
                name=modelname
            name=name+"Self"
            start = time.time()
            now = datetime.now()
            start_timeDistil = now.strftime("%H:%M:%S")

            newmodel=predictimagewithdistillationGlobal(scaling,minmax,f,test,model, modelType,type, XTrain, YTrain, name,indexes,R,Normalize,Mean,Stdev )
            end = time.time()
            now = datetime.now()
            end_timeDistil = now.strftime("%H:%M:%S")
            print("Time spent for self-training the classification model with R=",R, (end - start), file=f)
            print("Start Time self-training",start_timeDistil, file=f)
            print("End Time self-training", end_timeDistil, file=f)
            print("Start Time self-training", start_timeDistil)
            print("End Time self-training", end_timeDistil)
            namepickle = "Self"+name+"Model_R"+str(R)+"_fold"+str(fold)
            if (len(indexes) > 0):
                namepickle = namepickle + "_Indexes"
            pickle.dump(newmodel, open(namepickle+".pickle.dat", "wb"))


if __name__ == "__main__":


    start = 0
    middle = 78
    end = 94

    train=np.arange(start, middle, 1)
    test = np.arange(middle, end, 1)
    type=0
    Normalize=False
    selectedindexes=[]
    indexes = [
        # dal doc
        'CCCI', 'CHLGREEN', 'LCI', 'NDRE2', 'CVI', 'GDVI', 'GLI', 'GNDVI', 'NDVI', 'NG', 'NGDRI', 'BNDVI',
        'SRBlueRed',
        'GI', 'PBI', 'CI', 'PGR', 'DSWI', 'VMI', 'NDWI', 'SRSWIR', 'RDI',
        # from #https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
        'GNDVI', 'EVI', 'SAVI', 'NDMI', 'MSI', 'GCI', 'NBR', 'BSI',  # 'NDWI2',
        'ARVI', 'SIPI',
        ##https://www.mdpi.com/1424-8220/22/19/7440
        'NGRDI', 'DWSI',
        'DRS', 'ND790', 'NDVI690', 'GNDVIhyper', 'RENDVI1', 'RI',
        ##https://www.mdpi.com/2072-4292/14/13/3135
        'CLRE', 'GEMI', 'MCARI', 'MSAVI', 'NDREI2', 'NDRS',
        # from https://www.mdpi.com/2072-4292/12/21/3634;
        # from https://www.sciencedirect.com/science/article/pii/S0303243421000428?via%3Dihub
        # from DWSIBis https://www.mdpi.com/2072-4292/14/23/6105
        # from https://www.sciencedirect.com/science/article/pii/S0034425720306131?via%3Dihub#f0010
        'SLAVI', 'TVI', 'LAIGreen',  # 'NDVI550_650',
        'TCW', 'DWSIBis', 'NMDI', 'DRSBis']

    print("*************FOLD****************")
    fold=1

    print(">>>>>SVM")
    modelType=2 #SVM
    scaling = True
    selectedindexes = []
    classification(selectedindexes, train, test, fold, scaling, modelType)
    selectedindexes =['NMDI','MCARI','NGDRI']
    print(selectedindexes)
    classification(selectedindexes,train,test,fold,scaling,modelType)
    scaling = False

    print(">>>>>RF")
    scaling = False
    selectedindexes = []
    modelType = 1  # RF
    classification(selectedindexes, train, test, fold,scaling, modelType)
    selectedindexes = ['NMDI', 'MCARI', 'NGDRI']

    print(selectedindexes)
    classification(selectedindexes, train, test, fold, scaling, modelType)
    
    print(">>>>>XGBoost")
    scaling = False
    selectedindexes = []
    modelType = 0  # XGBoost
    classification(selectedindexes, train, test, fold, scaling, modelType)
    selectedindexes = ['NMDI', 'MCARI', 'NGDRI']

    print(selectedindexes)
    classification(selectedindexes, train, test, fold, scaling, modelType)

    ### fold 6
    print("*************END FOLD****************")