import numpy as np
import pickle

from sklearn.metrics import roc_auc_score

from loading import loaddataset, loaddataset_of_indexes, concatenatedataset

if __name__ == "__main__":
    path="D:/Progetti di ricerca/2022/SWIFTT@EUSPSA/wp1/bark beetle attacks in France/finalDB/risultati per  paper/div 10000 SPECTRAL + NMDI, MCARI, NGDRI/rf/"
    start = 0
    middle = 78
    end = 94
    train = np.arange(start, middle, 1)
    test = np.arange(middle, end, 1)
    type = 0
    Normalize = False
    selectedindexes = []
    fold=1

    with open("AUCROC" +str(fold)+ ".txt", "w") as f:
        for R in np.arange(1,2,1):

            selectedindexes = []
            X, Y, Mean, Stdev = loaddataset(test, type, Normalize)  # load original data):

            model_file_name = "RFModel_R" + str(R) + "_fold" + str(fold) + ".pickle"
            print(model_file_name)
            print(model_file_name, file=f)
            xgb_model_loaded = pickle.load(open(path + model_file_name + ".dat", "rb"))
            rocauc1=roc_auc_score(Y, xgb_model_loaded.predict_proba(X)[:, 1])
            print("AUCROC1=",rocauc1,)
            print("AUCROC1=",rocauc1,file=f)
            model_file_name = "SelfRFSelfModel_R" + str(R) + "_fold" + str(fold) + ".pickle"
            print(model_file_name)
            print(model_file_name, file=f)
            xgb_model_loaded = pickle.load(open(path + model_file_name + ".dat", "rb"))
            rocauc1 = roc_auc_score(Y, xgb_model_loaded.predict_proba(X)[:, 1])            
            print("AUCROC1=", rocauc1)
            print("AUCROC1=", rocauc1,  file=f)

            selectedindexes = ['NMDI', 'MCARI', 'NGDRI']

            XIndexes, YIndexes, MeanIndexes, StdevIndexes = loaddataset_of_indexes(test,selectedindexes,Normalize)
            X = concatenatedataset(X, XIndexes)
            model_file_name = "RF_IndexesModel_R" + str(R) + "_fold" + str(fold) + "_Indexes.pickle"
            print(model_file_name)
            print(model_file_name, file=f)
            xgb_model_loaded = pickle.load(open(path + model_file_name + ".dat", "rb"))
            rocauc1 = roc_auc_score(Y, xgb_model_loaded.predict_proba(X)[:, 1])

            print("AUCROC1=", rocauc1)
            print("AUCROC1=", rocauc1, file=f)
            model_file_name = "SelfRF_IndexesSelfModel_R" + str(R) + "_fold" + str(fold) + "_Indexes.pickle"
            print(model_file_name)
            print(model_file_name, file=f)
            xgb_model_loaded = pickle.load(open(path + model_file_name + ".dat", "rb"))
            rocauc1 = roc_auc_score(Y, xgb_model_loaded.predict_proba(X)[:, 1])

            print("AUCROC1=", rocauc1)
            print("AUCROC1=", rocauc1, file=f)
