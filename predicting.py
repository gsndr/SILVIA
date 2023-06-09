from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tifffile
import imageio

from correction import spatialCorrection
from learning import xgbfun, RFfun, SVMfun
from loading import loadimage, preprocessImage, generateindexes, loaddataset_labeledwithpriedictions, \
    normalizeTransform, concatenatedataset

seed=42
np.random.seed(seed)
def predictimage(minMaxScaler,scaler,file,test,model, typeImage=0, modelType='XGB', indexes=[], R=0,Normalize=False,Mean=[],Stdev=[]):

    firstImage=True
    for i in test:
        if firstImage==True:
            start=i
            maskname = 'masks/mask_' + str(i) + '.tif'
            predicted = 'labels/_' + modelType + 'label' + "_" + str(i) + "_" + str(R) + '.jpg'
            size = 0
            matrix = np.zeros((len(test), 4)) ##AA tocheck
            # print("#####Evaluation ", start, "-", end, "R=", R)
            try:
                image = loadimage(i, typeImage)
                d = image.shape
                size = size + (d[0] * d[1])
                X = preprocessImage(image)
                if (len(indexes) > 0):
                    Xindexes = generateindexes(X, indexes)  # genera indici
                    if (Normalize == True):
                        Xindexes = normalizeTransform(Xindexes, Mean, Stdev)
                    X = concatenatedataset(X, Xindexes)

                if(minMaxScaler==True):
                    X=scaler.transform(X)
                #mask = tifffile.imread(maskname)
                try:
                    mask = tifffile.imread(maskname)
                except OSError as err:
                    filename = 'images/geojson_' + str(i) + '.tif'
                    image = tifffile.imread(filename)
                    d = image.shape
                    data = image.reshape(d[0] * d[1], d[2])  # rehape row by row
                    Y = data[:, 13]
                    Y = np.nan_to_num(Y)
                    Y=np.reshape(Y,(d[0],d[1]))
                    mask= Y*255
                totalY = mask.reshape(d[0] * d[1], -1)  # rehape row by row
                np.place(totalY, totalY == 255, 1)
                totalpredY = model.predict(X)
                totalpredY = totalpredY.reshape(d[0], d[1])
                if (R > 0):
                    # print("***********Spatial correction with R:", R)
                    totalpredY = spatialCorrection(totalpredY, R)
                imageio.imwrite(predicted, (totalpredY * 255).astype(np.uint8))
                totalpredY = totalpredY.reshape(d[0] * d[1], -1)
                cm = confusion_matrix(totalY, totalpredY, labels=model.classes_.tolist())
                # print("***********Confusion matrix")
                # print(cm)
                # print("***********Classification Report  set")
                # print(classification_report(totalY, totalpredY, labels=model.classes_.tolist()))
                # tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
                matrix[i - start, 0], matrix[i - start, 1], matrix[i - start, 2], matrix[i - start, 3] = cm.ravel()
                firstImage=False
            except OSError as err:
                print("OS error:", err)
        else:
            filename = 'images/geojson_' + str(i) + '.tif'
            maskname = 'masks/mask_' + str(i) + '.tif'
            predicted = 'labels/_'+modelType+'label' + "_"+str(i) + "_"+str(R) +'.jpg'
            try:
                #image = tifffile.imread(filename)
                image=loadimage(i,typeImage)
                #print("Evaluate:", filename)
                d = image.shape
                size = size + (d[0] * d[1])
                X =preprocessImage(image)
                if(len(indexes)>0):
                    Xindexes = generateindexes(X, indexes)  # genera indici
                    if (Normalize == True):
                        Xindexes = normalizeTransform(Xindexes, Mean, Stdev)
                    X=concatenatedataset(X, Xindexes)
                if (minMaxScaler == True):
                    X = scaler.transform(X)
                #mask = tifffile.imread(maskname)
                try:
                    mask = tifffile.imread(maskname)
                except OSError as err:
                    filename = 'images/geojson_' + str(i) + '.tif'
                    image = tifffile.imread(filename)
                    d = image.shape
                    data = image.reshape(d[0] * d[1], d[2])  # rehape row by row
                    Y = data[:, 13]
                    Y = np.nan_to_num(Y)
                    Y=np.reshape(Y,(d[0],d[1]))
                    mask= Y*255
                Y = mask.reshape(d[0] * d[1], -1)  # rehape row by row
                np.place(Y, Y == 255, 1)
                totalY = np.concatenate((totalY, Y), axis=0)
                #print(filename, maskname)
                Y_pred = model.predict(X)
                Y_pred = Y_pred.reshape(d[0], d[1])
                if( R>0):
                    #print("***********Spatial correction with R:", R)
                    Y_pred=spatialCorrection(Y_pred,R)
                imageio.imwrite(predicted, (Y_pred*255).astype(np.uint8))
                Y_pred = Y_pred.reshape(d[0]*d[1],-1)
                totalpredY = np.concatenate((totalpredY, Y_pred), axis=0)
                #print("***********Confusion matrix")
                cm = confusion_matrix(Y, Y_pred, labels=model.classes_.tolist())
                matrix[i-start,0], matrix[i-start,1], matrix[i-start,2], matrix[i-start,3] = cm.ravel()
                #print(cm)
                #print("***********Classification Report  set")
                #print(classification_report(Y, Y_pred, labels=model.classes_.tolist()))


            except OSError as err:
                print("OS error:", err)
    print("***********Confusion matrix on ", size , "pixels")
    print("***********Confusion matrix on ", size, "pixels",file=file)
    cm = confusion_matrix(totalY, totalpredY, labels=model.classes_.tolist())
    print(cm)
    print(cm,file=file)
    print("***********Classification Report  set")
    print("***********Classification Report  set",file=file)
    print(classification_report(totalY, totalpredY, labels=model.classes_.tolist()))
    print(classification_report(totalY, totalpredY, labels=model.classes_.tolist()),file=file)
    print("#####End Evaluation")
    print("#####End Evaluation", file=file)
    print(matrix.shape)

    import pandas as pd
    end=test[len(test)-1]
    pd.DataFrame(matrix).to_csv('labels/_'+modelType+'ReportMatrix' + "_"+str(start) + "_"+str(end)+"_"+str(R) +'.csv')
    return



# refine prediction of xgb with self learning of xgb on training set +labele testing set (si usa un solo tipo di feature in entrambe le fasi

def predictimagewithdistillationGlobal(scaling,minmax,file,test,model,modelType, typeImage, XTrain, YTrain, modelname='XGB', indexes=[],R=0,Normalize=False,Mean=[],Stdev=[]):

    i=test[0]
    maskname = 'masks/mask_' + str(i) + '.tif'
    predicted = 'labels/_'+modelname+'label' + "_"+str(i) + "_"+str(R) +'.jpg'
    size=0

    XTest,YTest=loaddataset_labeledwithpriedictions(scaling,minmax,test, typeImage, model,indexes,R,Normalize,Mean,Stdev)
    XTrain = np.concatenate((XTrain, XTest), axis=0)
    YTrain = np.concatenate((np.reshape(YTrain,-1), YTest), axis=0)
    #train a new model on the training set extended wit the predicted labesl (in the self training mode)
    if(modelType==0):
        newmodel,score,par1,par2,par3= xgbfun(XTrain, YTrain)
    elif(modelType==1):
        newmodel,score, par1,par2,par3,par4 = RFfun(XTrain, YTrain)
    elif(modelType==2):
        newmodel, score, par1, par2, par3 = SVMfun(XTrain, YTrain)
    predictimage(scaling,minmax,file,test, newmodel, typeImage, modelname,indexes, R)
    return newmodel


