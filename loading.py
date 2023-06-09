import imageio
import numpy as np
import tifffile
from ConvertUtility import imread
from Sentinel2Index import ATSAVI, AFRI1600, CCCI, CHLGREEN, LCI, NDRE2, CVI, GDVI, GLI, GNDVI, NDVI, NG, NGDRI, BNDVI, \
    SRBlueRed, CIGREEN, GI, PBI, CI, PGR, DSWI, VMI, LWCI, NDWI, SRSWIR, RDI, EVI, AVI2, SAVI, GCI, BSI, NDWI2, ARVI, \
    SIPI, NDMI, MSI, NBR, DWSI, NDRE2Bis, DRS, ND790, NDVI690, GNDVIhyper, RENDVI1, RI, CLRE, GEMI, MCARI, MSAVI, \
    NDREI2, NDRS, NRVI, SLAVI, TVI, LAIGreen, NDVI550_650, TCW, DWSIBis, NMDI, DRSBis
import pandas as pd

# Preprocess image to remove band 10 and 11
from correction import spatialCorrection

seed=42
np.random.seed(seed)

def preprocessImage(image): #remove 10,11 add ndvi


    d = image.shape
    XTrain = image.reshape(d[0] * d[1], d[2])  # rehape row by row
    if(d[2]>10):
         XTrain = np.delete(XTrain, 11, 1)
         XTrain = np.delete(XTrain, 10, 1)

    XTrain=XTrain/10000
    return  XTrain




def generateindexes(XTrain, indexes=[]):
    rows = XTrain.shape[0]
    newXTrain=np.zeros((rows,1))
    XTrain1=XTrain #value / 10000



    for index, name in enumerate(indexes):
        #print(name)
        if name == "ATSAVI":
            band = ATSAVI(XTrain1)
        elif name == "AFRI1600":
            band = AFRI1600(XTrain1)
        elif name == "ARVI":
            band = ARVI(XTrain1)
        elif name == "AVI2":
            band = AVI2(XTrain1)
        elif name == "BSI":
            band = BSI(XTrain1)
        elif name == "BNDVI":
            band = BNDVI(XTrain1)
        elif name == "CCCI":
            band = CCCI(XTrain1)
        elif name == "CHLGREEN":
            band = CHLGREEN(XTrain1)
        elif name == "CI":
            band = CI(XTrain1)
        elif name == "CIGREEN":
            band = CIGREEN(XTrain1)
        elif name == "CLRE":
            band = CLRE(XTrain1)
        elif name == "CVI":
            band = CVI(XTrain1)
        elif name== "DRS":
            band = DRS(XTrain1)
        elif name== "DRSBis":
            band = DRSBis(XTrain1)
        elif name == "DSWI":
            band = DSWI(XTrain1)
        elif name == "DWSI":
            band = DWSI(XTrain1)
        elif name == "DWSIBis":
            band = DWSIBis(XTrain1)
        elif name == "EVI":
            band = EVI(XTrain1)
        elif name == "GCI":
            band = GCI(XTrain1)
        elif name == "GDVI":
            band = GDVI(XTrain1)
        elif name== "GEMI":
            band= GEMI(XTrain1)
        elif name == "GI":
            band = GI(XTrain1)
        elif name == "GLI":
            band = GLI(XTrain1)
        elif name == "GNDVI":
            band = GNDVI(XTrain1)
        elif name == "GNDVIhyper":
            band = GNDVIhyper(XTrain1)
        elif name=='LAIGreen':
            band = LAIGreen(XTrain1)
        elif name == "LCI":
            band = LCI(XTrain1)
        elif name == "LWCI":
            band = LWCI(XTrain1)
        elif name == "MCARI":
            band= MCARI(XTrain1)
        elif name=="MSAVI":
            band = MSAVI(XTrain1)
        elif name == "MSI":
            band = MSI(XTrain1)
        elif name == "NBR":
            band = NBR(XTrain1)
        elif name== "NDRS":
            band= NDRS(XTrain1)
        elif name=="ND790":
            band = ND790(XTrain1)
        elif name=="NDREI2":
            band= NDREI2(XTrain1)
        elif name=='NDVI690':
            band = ND790(XTrain1)
        elif name == "NDMI":
            band = NDVI690(XTrain1)
        elif name=='NDVI550_650':
            band=NDVI550_650(XTrain1)
        elif name == "NDWI":
            band = NDWI(XTrain1)
        elif name == "NDWI2":
            band = NDWI2(XTrain1)
        elif name == "NDRE2":
            band = NDRE2(XTrain1)
        elif name == "NDRE2Bis":
            band = NDRE2Bis(XTrain1)
        elif name == "NDVI":
            band = NDVI(XTrain1)
        elif name == "NG":
            band = NG(XTrain1)
        elif name == "NGDRI":
            band = NGDRI(XTrain1)
        elif name== "NMDI":
            band= NMDI(XTrain1)
        elif name == "NRVI":
            band =NRVI(XTrain1)
        elif name == "PBI":
            band = PBI(XTrain1)
        elif name == "PGR":
            band = PGR(XTrain1)
        elif name == "RDI":
            band = RDI(XTrain1)
        elif name=="RENDVI1":
            band = RENDVI1(XTrain1)
        elif name=="RI":
            band = RI(XTrain1)
        elif name == "SAVI":
            band = SAVI(XTrain1)
        elif name == "SIPI":
            band = SIPI(XTrain1)
        elif name == "SLAVI":
            band = SLAVI(XTrain1)
        elif name == "SRBlueRed":
            band = SRBlueRed(XTrain1)
        elif name == "SRSWIR":
            band = SRSWIR(XTrain1)
        elif name == "TCW":
            band = TCW(XTrain1)
        elif name == "TVI":
            band = TVI(XTrain1)
        elif name == "VMI":
            band = VMI(XTrain1)
        else:
            print(name, "missing")

        band = np.reshape(band, (rows, 1))
        newXTrain = np.concatenate((newXTrain, band), axis=1)
    newXTrain=np.delete(newXTrain,0,1)
    return newXTrain

# load image from tiff file (sentinel data)
def loadSentinelImage(i):
    filename = 'images/geojson_' + str(i) + '.tif'
    image = tifffile.imread(filename)
    return image

def loadGOimage(i):
    filename = "txtimages/geojson_" + str(i) + "ScaleGOS2Scale.txt"
    image = imread(filename)
    return image

def loadimage(i,type=0):
    if (type==0): #sentinel #0 indexes type 3
        return loadSentinelImage(i)
    elif type==1: #getis and ord
        return loadGOimage(i)
    else: #sentinen+ getis and ord
        image1=loadSentinelImage(i)
        image2 = loadGOimage(i)
        image = np.concatenate((image1, image2), axis=2)
        return image

#load a dataset from files type= 0 tiff, type=1 go, type=2 tiff+ go
def loaddataset(train,type=0, Normalize=False):
    firstImage=True
    #print(maskname)
    for i in train:
        if (firstImage==True):
            Mean = []
            Stdev = []
            maskname = 'masks/mask_' + str(i) + '.tif'
            try:
                image = loadimage(i, type)
                d = image.shape
                XTrain = preprocessImage(image)
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
                YTrain = mask.reshape(d[0] * d[1], -1)  # rehape row by row
                np.place(YTrain, YTrain == 255, 1)
                firstImage=False
            except OSError as err:
                print("OS error:", err)
        else:
            maskname='masks/mask_'+str(i)+'.tif'
            try:
                image=loadimage(i,type)
                d=image.shape
                X = preprocessImage(image)
                #mask = tifffile.imread(maskname)
                try:
                    mask = tifffile.imread(maskname)
                except OSError as err: ##nel caso vanno presi dalle nuove immagini
                    filename = 'images/geojson_' + str(i) + '.tif'
                    image = tifffile.imread(filename)
                    d = image.shape
                    data = image.reshape(d[0] * d[1], d[2])  # rehape row by row
                    Y = data[:, 13]
                    Y = np.nan_to_num(Y)
                    Y = np.reshape(Y, (d[0], d[1]))
                    mask = Y * 255
                Y=mask.reshape(d[0]*d[1],-1) # rehape row by row
                np.place(Y, Y == 255, 1)
                #print(maskname)
                XTrain=np.concatenate((XTrain,X), axis=0)
                YTrain = np.concatenate((YTrain, Y),axis=0)
            except OSError as err:
                print("OS error:", err)
    if(Normalize==True):
        XTrain,Mean,Stdev = normalize(XTrain, YTrain, 0)
    return XTrain,YTrain,Mean,Stdev

#load the dataset labeled with predictions provided by model
#if indexes is not empty then type=0 or type=2, and the dataset is composed of indexes
def loaddataset_labeledwithpriedictions(scaling,minmax,train, type, model,indexes=[],R=0, Normalize=False,MeanIndexes=[], StdevIndexes=[]):
    firstImage=True
    # print(maskname)
    for i in train:
        if(firstImage==True):
            try:
                maskname = 'masks/mask_' + str(i) + '.tif'
                image = loadimage(i, type)
                d = image.shape
                XTrain = preprocessImage(image)
                if (len(indexes) > 0):
                    XTrainIndexes = generateindexes(XTrain, indexes)
                    if (Normalize == True):
                        XTrainIndexes = normalizeTransform(XTrainIndexes, MeanIndexes, StdevIndexes)
                    XTrain = concatenatedataset(XTrain, XTrainIndexes)
                if(scaling==True):
                    XTrain=minmax.transform(XTrain)
                YTrain = model.predict(XTrain)
                YTrain = YTrain.reshape(d[0], d[1])
                if (R > 0):
                    # print("***********Spatial correction with R:", R)
                    Y = spatialCorrection(YTrain, R)
                YTrain = YTrain.reshape((d[0] * d[1]))
                firstImage=False
            except OSError as err:
                print("OS error:", err)
        else:
            maskname = 'masks/mask_' + str(i) + '.tif'
            try:
                image = loadimage(i, type)
                d = image.shape
                X = preprocessImage(image)
                if (len(indexes) > 0):
                    XIndexes = generateindexes(X, indexes)
                    if (Normalize == True):
                        XIndexes = normalizeTransform(XIndexes, MeanIndexes, StdevIndexes)
                    X = concatenatedataset(X, XIndexes)
                if (scaling == True):
                    X = minmax.transform(X)
                Y = model.predict(X)
                Y = Y.reshape(d[0], d[1])
                if (R > 0):
                    # print("***********Spatial correction with R:", R)
                    Y = spatialCorrection(Y, R)
                Y=Y.reshape((d[0]* d[1]))
                # print(maskname)
                XTrain = np.concatenate((XTrain, X), axis=0)
                YTrain = np.concatenate((YTrain, Y), axis=0)
            except OSError as err:
                print("OS error:", err)

    return XTrain, YTrain


#create a dataset with indexes
def loaddataset_of_indexes(train, indexes=[],Normalize=False):
    firstImage=True
    for i in train:
        if(firstImage==True):
            try:
                Mean = []
                Stdev = []
                maskname = 'masks/mask_' + str(i) + '.tif'
                image = loadimage(i)  # type=0
                d = image.shape
                # print(maskname)
                XTrain = preprocessImage(image)
                XTrain = generateindexes(XTrain, indexes)
                #mask = tifffile.imread(maskname)
                try:
                    mask = tifffile.imread(maskname)
                except OSError as err: ##nel caso vanno presi dalle nuove immagini
                    filename = 'images/geojson_' + str(i) + '.tif'
                    image = tifffile.imread(filename)
                    d = image.shape
                    data = image.reshape(d[0] * d[1], d[2])  # rehape row by row
                    Y = data[:, 13]
                    Y = np.nan_to_num(Y)
                    Y = np.reshape(Y, (d[0], d[1]))
                    mask = Y * 255
                YTrain = mask.reshape(d[0] * d[1], -1)  # rehape row by row
                np.place(YTrain, YTrain == 255, 1)
                firstImage=False
            except OSError as err:
                print("OS error:", err)
        else:
            maskname = 'masks/mask_' + str(i) + '.tif'
            try:
                #print(maskname)
                image = loadimage(i)
                d = image.shape
                X = preprocessImage(image)
                X = generateindexes(X, indexes)
                #mask = tifffile.imread(maskname)
                try:
                    mask = tifffile.imread(maskname)
                except OSError as err: ##nel caso vanno presi dalle nuove immagini
                    filename = 'images/geojson_' + str(i) + '.tif'
                    image = tifffile.imread(filename)
                    d = image.shape
                    data = image.reshape(d[0] * d[1], d[2])  # rehape row by row
                    Y = data[:, 13]
                    Y = np.nan_to_num(Y)
                    Y = np.reshape(Y, (d[0], d[1]))
                    mask = Y * 255
                Y = mask.reshape(d[0] * d[1], -1)  # rehape row by row
                np.place(Y, Y == 255, 1)
                XTrain = np.concatenate((XTrain, X), axis=0)
                YTrain = np.concatenate((YTrain, Y), axis=0)
            except OSError as err:
                print("OS error:", err)
    #XTrain[np.isnan(XTrain)] = 0
    if(Normalize==True):
        XTrain,Mean,Stdev = normalize(XTrain, YTrain, 0)
    return XTrain,YTrain,Mean,Stdev

#concatenate two datasets along axis X
def concatenatedataset(XTrain1, XTrain2):
    return np.concatenate((XTrain1, XTrain2), axis=1)

#normalize wrt a specific class

def normalize(XTrain, YTrain, category=0): #DOI: 10.15287/afr.2015.388
    df = pd.DataFrame(YTrain,columns=list('A'))
    indexes=(df.index[df['A'] == category].tolist())
    XTrainSel=XTrain[indexes,:]
    print("Normalize on ", XTrainSel.shape[0], "examples")
    mean=np.mean(XTrainSel, axis=0)
    stdev=np.std(XTrainSel, axis=0)
    XTrain=np.where(stdev==0.0,0.0,(XTrain-mean)/stdev)
    return XTrain,mean,stdev

def normalizeTransform(XTrain, mean, stdev):
    XTrain=np.where(stdev==0.0,0,(XTrain-mean)/stdev)
    return XTrain