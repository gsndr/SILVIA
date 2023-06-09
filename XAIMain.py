import pickle

import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from loading import concatenatedataset, loaddataset, loaddataset_of_indexes
import pandas as pd
import dalex as dx

seed = 42


def shap_explainer(modelType, trainXdata, trainingYdata, datatoexplain, Ytoexplain, model, filetosave):
    if (modelType == 0):  # XGBoost
        XTraining, XValidation, YTraining, YValidation = train_test_split(trainXdata, trainingYdata,
                                                                          stratify=trainingYdata, train_size=500,
                                                                          random_state=seed)
        print('Shap XTraining shape : ', XTraining.shape)
        explainer = shap.KernelExplainer(model.predict_proba, XTraining, algorithm='permutation')  # this for XGB and RF
        print(' Computing Explanations...')
        shap_values = explainer.shap_values(datatoexplain, nsamples=100)
    elif (modelType == 1):  # RF
        XTraining, XValidation, YTraining, YValidation = train_test_split(trainXdata, trainingYdata,
                                                                          stratify=trainingYdata, train_size=500,
                                                                          random_state=seed)
        print('Shap XTraining shape : ', XTraining.shape)
        explainer = shap.KernelExplainer(model.predict_proba, XTraining, algorithm='permutation')  # this for XGB and RF
        print(' Computing Explanations...')
        shap_values = explainer.shap_values(datatoexplain, nsamples=100)
    elif (modelType == 2):  # SVM
        '''
        XTraining, XValidation, YTraining, YValidation = train_test_split(trainXdata, trainingYdata,
                                                                          stratify=trainingYdata, train_size=500,
                                                                          random_state=seed)
        print('Shap XTraining shape : ', XTraining.shape)
        explainer = shap.KernelExplainer(model.predict_proba, XTraining, algorithm='permutation')  # this for XGB and RF
        print(' Computing Explanations...')
        shap_values = explainer.shap_values(datatoexplain, nsamples=100)
        '''
        # shap values computed on  blocks of testing data due to efficiency issus)
        XTraining, XValidation, YTraining, YValidation = train_test_split(trainXdata, trainingYdata,
                                                                          stratify=trainingYdata, train_size=100,
                                                                          random_state=seed)
        print("count y=1", np.sum(YTraining))
        print('Shap XTraining shape : ', XTraining.shape)
        explainer = shap.KernelExplainer(model.predict_proba, XTraining, algorithm='permutation')  # this is for sVM
        print(' Computing Explanations with 100'
              '...')

        part = 1

        examples, features = datatoexplain.shape
        print("Examples to process:", examples)
        while (examples > 0):  ##ci sono ancora dati da spiegare
            Xdatatoexplain, XValdatatoexplain, Ydatatoexplain, YValdatatoexplain = train_test_split(datatoexplain,
                                                                                                    Ytoexplain,
                                                                                                    stratify=Ytoexplain,
                                                                                                    train_size=5000,
                                                                                                    random_state=seed)

            y = np.asarray(Ydatatoexplain)
            np.save("tempY_" + str(part), y)  # save GT Y
            print("part:", part, " count y=1:", np.sum(Ydatatoexplain))
            print(Xdatatoexplain.shape)
            shap_values = explainer.shap_values(Xdatatoexplain, nsamples=50)  # 50
            xai = np.asarray(shap_values)
            print('xai shape: ', xai.shape)
            # print(xai)
            np.save(filetosave + "_" + str(part), xai)  # save shape
            datatoexplain = XValdatatoexplain
            Ytoexplain = YValdatatoexplain
            examples, features = datatoexplain.shape
            print("Remaining examples to process:", examples)
            part = part + 1

    print(' Computing Explanations...')
    # shap_values = explainer.shap_values(datatoexplain, nsamples=100)
    xai = np.asarray(shap_values)
    print('xai shape: ', xai.shape)
    np.save(filetosave, xai)

    #y = np.asarray(Ydatatoexplain)
    #print("shape y", y.shape)
    #np.save("tempY", y)
    return xai


def xaishap(modelType, model, train, test, indexes, filetosave, type=0, Normalize=False, scaling=False):
    Mean = []
    Stdev = []
    # load training set
    print('Scaling:', scaling)
    XTrainOri, YTrain, MeanTrainOri, StdevTrainOri = loaddataset(train, type, Normalize)  # load original data):
    if (len(indexes) > 0):
        XTrainIndexes, YTrainIndexes, MeanTrainIndexes, StdevTrainIndexes = loaddataset_of_indexes(train, indexes,
                                                                                                   Normalize)
        XTrain = concatenatedataset(XTrainOri, XTrainIndexes)
    else:
        XTrain = XTrainOri
    minmax = MinMaxScaler()
    if (scaling == True):
        XTrain = minmax.fit_transform(XTrain)
    # load testing set to explain
    XTestOri, YTest, MeanTestOri, StdevTestOri = loaddataset(test, type, Normalize)  # load original data):
    if (len(indexes) > 0):
        XTestIndexes, YTestIndexes, MeanTestIndexes, StdevTestIndexes = loaddataset_of_indexes(test, indexes, Normalize)
        XTest = concatenatedataset(XTestOri, XTestIndexes)
    else:
        XTest = XTestOri
    if (scaling == True):
        XTest = minmax.transform(XTest)
    xai = shap_explainer(modelType, XTrain, YTrain, XTest, YTest, model, filetosave)
    print(xai.size)


def elaborate_shap(test, indexes, filetoload, type=0, Normalize=False):
    xai = np.load(filetoload)
    n, m, k = xai.shape
    print("Shape", n, m, k)
    xai0 = xai[0, :, :]
    xai1 = xai[1, :, :]
    # load testing set to explain
    XTest, YTest, MeanTest, StdevTest = loaddataset(test, type, Normalize)  # load original data):
    independentList = ['Coastal Aerosol', 'Blue', 'Green', 'Red', 'RED1', 'RED3', 'NIR', 'Water Vapor', 'SWIR 1',
                       'SWIR 2']
    independentList = independentList + indexes
    print(independentList)
    df = pd.DataFrame(YTest, columns=list('A'))
    indexes0 = (df.index[df['A'] == 0.0].tolist())
    xai0 = xai0[indexes0, :]
    indexes1 = (df.index[df['A'] != 0.0].tolist())
    xai1 = xai1[indexes1, :]
    rank0 = np.average(xai0, axis=0)
    dict_from_list0 = {k: v for k, v in zip(independentList, rank0)}
    sorteddict_from_list0 = sorted(dict_from_list0.items(), key=lambda x: x[1], reverse=True)
    dict_from_list0 = dict(sorteddict_from_list0)

    rank1 = np.average(xai1, axis=0)
    dict_from_list1 = {k: v for k, v in zip(independentList, rank1)}
    sorteddict_from_list1 = sorted(dict_from_list1.items(), key=lambda x: x[1], reverse=True)
    dict_from_list1 = dict(sorteddict_from_list1)
    print("Rank Class 0", dict_from_list0)
    print("Rank Class 1", dict_from_list1)
    return dict_from_list0, dict_from_list1


def dalex_explainer(XTrain, YTrain, model):
    # spiego intero db
    d = YTrain.shape
    YTrain = np.reshape(YTrain, -1)
    explainer = dx.Explainer(model, XTrain, YTrain)
    explanation = explainer.model_parts(random_state=42)
    variable_importance = pd.DataFrame(explanation.result)
    print(variable_importance)
    return variable_importance


def dalex(model, train, indexes, type=0, Normalize=False):
    XTrainOri, YTrain, MeanTrainOri, StdevTrainOri = loaddataset(train, type, Normalize)  # load original data):
    if (len(indexes) > 0):
        XTrainIndexes, YTrainIndexes, MeanTrainIndexes, StdevTrainIndexes = loaddataset_of_indexes(train, indexes,
                                                                                                   Normalize)
        XTrain = concatenatedataset(XTrainOri, XTrainIndexes)
    else:
        XTrain = XTrainOri
    xai = dalex_explainer(XTrain, YTrain, model)


if __name__ == "__main__":
    path = ""
    start = 0
    middle = 78
    end = 94

    train = np.arange(start, middle, 1)
    test = np.arange(middle, end, 1)
    type = 0
    Normalize = False

    fold = 1
    #####code to compute local shap tensor on testing set
    modelType=0 #XGBoost
    # compute shap R=1, self, indenxes
    R = 1

    selectedindexes = ['NMDI', 'MCARI', 'NGDRI']
    #XGB
    # SelfXGB_IndexesSelfModel_R1_fold1_Indexes.pickle
    model_file_name = path + "SelfXGB_IndexesSelfModel_R" + str(R) + "_fold" + str(fold) + "_Indexes.pickle"
    print(model_file_name)
    xgb_model_loaded = pickle.load(open(path + model_file_name + ".dat", "rb"))
    fileToSaveXAI = "SelfXGB_IndexesSelfModel_R" + str(R) + "_fold" + str(fold) + "_IndexesXAI"
    xai = xaishap(modelType,xgb_model_loaded, train, test, selectedindexes, fileToSaveXAI)

    #### code to produce global shap values
    with open("SHAPGLOBAL" + str(fold) + ".txt", "w") as f:
        R = 1
        selectedindexes = ['NMDI', 'MCARI', 'NGDRI']
        #SelfXGB_IndexesSelfModel_R1_fold1_Indexes.pickle
        fileToSaveXAI = path + "SelfXGB_IndexesSelfModel_R" + str(R) + "_fold" + str(fold) + "_IndexesXAI"
        global0, global1 = elaborate_shap(test, selectedindexes, fileToSaveXAI + ".npy")
        print(fileToSaveXAI, file=f)
        print(fileToSaveXAI)
        print("Class 0", file=f)
        print(global0, file=f)
        print("Class 1", file=f)
        print(global1, file=f)



