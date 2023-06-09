#B01 (0) B02(1) B03 (2) B04(3) B05(4) B07 (5) B08 (6) B09 (7) B011 (8) B012 (9)
#https://eo4geocourses.github.io/IGIK_Sentinel2-Data-and-Vegetation-Indices/#/
#https://www.indexdatabase.de/db/is.php?sensor_id=96
#https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/
#https://en.wikipedia.org/wiki/Sentinel-2
#selezione di indici https://www.mdpi.com/2072-4292/14/13/3135
#B01 (0) B02(1) B03 (2) B04(3) B05(4) B07 (5) B08 (6) B09 (7) B011 (8) B012 (9)
import numpy as np
seed=42
np.random.seed(seed)

band_dictionary = {'B01': 0, 'B02':1, 'B03' :2 , 'B04': 3, 'B05': 4, 'B07': 5, 'B08':6,
                   'B09': 7, 'B011':8, 'B012':9}
'''
band_dictionary = {'B01': 0, 'B02':1, 'B03' :2 , 'B04': 3, 'B05': 4,'B06' :5, 'B07': 6, 'B08':7, 'B08A':8,
                   'B09': 9, 'B011':10, 'B012':11, 'B013':12}
'''

def ATSAVI(XTrain): # 1.22 * (B08 - 1.22 * B04 - 0.03) / (1.22 * B08 + B04 - 1.22 * 0.03 + 0.08 * (1.0 + Math.pow(1.22, 2.0)));
    B08 = XTrain[:, band_dictionary['B08']]
    B04 = XTrain[:, band_dictionary['B04']]
    d=(1.22*B08+B04-1.22*0.03+0.08*(1+1.22**2))
    ATSAVI=np.where(d== 0, 0,1.22*(B08-1.22*B04-0.03)/d)
    return ATSAVI

def AFRI1600(XTrain): #B08 - 0.66 * B011 / (B08 + 0.66 * B011);
    B08 = XTrain[:, band_dictionary['B08']]
    B011 = XTrain[:, band_dictionary['B011']]
    d=(B08+0.66*B011)
    AFRI1600=np.where(d== 0, 0,B08-0.66*B011/d)
    return AFRI1600

def AFRI2100(XTrain): #	B08 - 0.5 * B012 / (B08 + 0.56 * B012);
    B08 = XTrain[:, band_dictionary['B08']]
    B012 = XTrain[:, band_dictionary['B012']]
    d=(B08+0.56*B012)
    AFRI2100=np.where(d==0, 0,B08-0.5*B012/d)
    return AFRI2100

def ALTERATION(XTrain): #	(B011/B012)
    B011 = XTrain[:, band_dictionary['B011']]
    B012 = XTrain[:, band_dictionary['B012']]
    ALTERATION=np.where(B012 == 0, 0,B011/B012)
    return ALTERATION

def ARI(XTrain): #	 1.0 / B03 - 1.0 / B05;
    B03 = XTrain[:, band_dictionary['B03']]
    B05 = XTrain[:, band_dictionary['B05']]
    ARI=np.where(B03 == 0 or B05==0, 0,(1/B03)-(1/B05))
    return ARI
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
def ARVI(XTrain): #		 ARVI = (NIR – (2 * Red) + Blue) / (NIR + (2 * Red) + Blue)
    B02 = XTrain[:, band_dictionary['B02']]
    B04 = XTrain[:, band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=(B08+(2*B04)+B02)
    ARVI2=np.where(d == 0, 0,(B08-(2*B04)+B02)/d)
    return ARVI2
#ARVI
def ARVI2(XTrain): #		 -0.18 + 1.17 * ((B08 - B04) / (B08 + B04));
    B04 = XTrain[:, band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=(B08+B04)
    ARVI2=np.where(d == 0, 0,-0.18+0.17*((B08-B04)/d))
    return ARVI2

def AVI(XTrain): #2.0 * B09 - B04;
    B04 = XTrain[:, band_dictionary['B04']]
    B09 = XTrain[:, band_dictionary['B09']]
    AVI = 2*B09-B04
    return AVI
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
def AVI2(XTrain): #•	AVI (Sentinel 2) = [B8 * (1 – B4)*(B8 – B4)]1/3
    B04 = XTrain[:, band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=(B08*(1-B04)*(B08-B04))
    AVI2 =np.where(d<=0, 0, np.power(d,1/3))
    return AVI2
#Normalized Difference NIR/Blue Blue-normalized difference vegetation index
def BNDVI(XTrain): #(B08 - B02) / (B08 + B02);
    B02 = XTrain[:, band_dictionary['B02']]
    B08 = XTrain[:, band_dictionary['B08']]
    d = (B08 + B02)
    BNDVI = np.where(d == 0, 0, (B08 - B02) / (d))
    return BNDVI
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
def BSI(XTrain): #(B11 + B4) – (B8 + B2) / (B11 + B4) + (B8 + B2)
    B02 = XTrain[:, band_dictionary['B02']]
    B04 = XTrain[:, band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    B011 = XTrain[:, band_dictionary['B011']]
    d = (B011+B04)
    BSI = np.where(d == 0, 0, d-(B08+B02)/d + (B08+B02))
    return BSI
def BWDRVI(XTrain): #		(0.1 * B08 - B02) / (0.1 * B08 + B02);
    B02 = XTrain[:, band_dictionary['B02']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=(0.1*B08+B02)
    BWDRVI=np.where(d == 0, 0,(0.1*B08-B02)/d)
    return BWDRVI
def BRI(XTrain): #			(1.0 / B03 - 1.0 / B05) / B08;
    B03 = XTrain[:, band_dictionary['B03']]
    B05 = XTrain[:, band_dictionary['B05']]
    B08 = XTrain[:, band_dictionary['B08']]
    cond= ((B03 == 0) | (B05==0) | (B08==0))
    BRI=np.where((cond==True), 0,(1.0 / B03 - 1.0 / B05) / B08)
    return BRI
#Canopy Chlorophyll Content Index
def CCCI(XTrain): #((B08 - B05) / (B08 + B05)) / ((B08 - B04) / (B08 + B04));ub.com/custom-scripts/sentinel-2/indexdb/id_224.js
    B04 = XTrain[:, band_dictionary['B04']]
    B05 = XTrain[:, band_dictionary['B05']]
    B08 = XTrain[:, band_dictionary['B08']]
    d1=(B08+B05)
    d2=(B08-B04)
    d3=(B08+B04)
    cond=(d1 == 0) | (d2==0) | (d3==0)
    CCCI = np.where((cond==True), 0,((B08-B05)/d1)/(d2/d3))
    return CCCI

def CARI(XTrain): #(B05 / B04) * (Math.sqrt(Math.pow(((B05 - B03) / 150.0 * 670.0 + B04 + (B03 - ((B05 - B03) / 150.0 * 550.0))), 2.0))) / (Math.pow(((B05 - B03) / Math.pow(150.0, 2.0) + 1.0), 0.5));
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    B05 = XTrain[:, band_dictionary['B05']]
    d=(np.pow(((B05 - B03) / np.pow(150.0, 2.0) + 1.0), 0.5))
    cond=((B04==0) | (d==0))
    CARI = np.where((cond==True), 0,(B05 / B04) * (np.sqrt(np.pow(((B05 - B03) / 150.0 * 670.0 + B04 + (B03 - ((B05 - B03) / 150.0 * 550.0))), 2.0))) / d)
    return CARI

def CARI2(XTrain): #(Math.abs(((B05 - B03) / 150.0 * B04 + B04 + B03 - (a * B03))) / Math.pow((Math.pow(a, 2.0) + 1.0), 0.5)) * (B05 / B04);
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    B05 = XTrain[:, band_dictionary['B05']]
    a = 0.496
    d=np.pow((np.pow(a, 2.0) + 1.0), 0.5)
    cond=((B04==0) | (d==0))
    CARI2 = np.where(cond==True, 0,(np.abs(((B05 - B03) / 150.0 * B04 + B04 + B03 - (a * B03))) / d) * (B05 / B04))
    return CARI2
#Chlorophyll Green	Chlgreen
def CHLGREEN(XTrain):  #Math.pow((B07 / B03), (-1.0));
    B03 = XTrain[:, band_dictionary['B03']]
    B07 = XTrain[:, band_dictionary['B07']]
    CHLGREEN=np.where(B07==0, 0,B03 / B07)
    return CHLGREEN
def Chlrededge(XTrain): # Math.pow((B07 / B05), (-1.0))
    B05 = XTrain[:, band_dictionary['B05']]
    B07 = XTrain[:, band_dictionary['B07']]
    Chlrededge = np.where((B05 == 0), 0,np.power((B07 / B05), (-1.0)));
    return Chlrededge

#Coloration Index
def CI(XTrain): # (B04 - B02) / B04;
    B02 = XTrain[:, band_dictionary['B02']]
    B04 = XTrain[:, band_dictionary['B04']]
    CI = np.where((B04 == 0), 0,(B04 - B02) / B04);
    return CI
#Chlorophyll index green
def CIGREEN(XTrain): #B08 / B03 - 1.0;
    B03 = XTrain[:, band_dictionary['B03']]
    B08 = XTrain[:, band_dictionary['B08']]
    CIGREEN=np.where((B03==0), 0,B08 / B03 - 1.0);
    return CIGREEN
def CIrededge(XTrain): # B08 / B05 - 1.0;
    B05 = XTrain[:, band_dictionary['B05']]
    B08 = XTrain[:, band_dictionary['B08']]
    CIrededge = np.where((B05 == 0), 0,B08 / B05 - 1.0);
    return CIrededge
#CLRE—Red-Edge Band Chlorophyll Index
#https://www.mdpi.com/2072-4292/14/13/3135
#formula trovata in Using the red-edge bands on Sentinel-2 for retrieving canopy chlorophyll and nitrogen content January 2012 J.G.P.W. CleversJ.G.P.W. CleversAnatoly GitelsonAnatoly Gitelson
def CLRE(XTrain): #B7/B5-1
    B05 = XTrain[:, band_dictionary['B05']]
    B07 = XTrain[:, band_dictionary['B07']]
    CLRE = np.where(B05 == 0, 0, (B07 / B05) -1)
    return CLRE
#Chlorophyll vegetation index
def CVI(XTrain): # B08 * B04 / Math.pow(B03, 2.0);
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    CVI = np.where((B03 == 0), 0, (((B08 * B04) / B03)/ B03))
    return CVI
def CTVI(XTrain): # (((B04 - B03) / (B04 + B03)) + 0.5) / Math.abs(((B04 - B03) / (B04 + B03)) + 0.5) * Math.sqrt(Math.abs((((B04 - B03) / (B04 + B03))) + 0.5))
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    d=(B04 + B03)
    CTVI = np.where((d == 0), 0,(((B04 - B03) / d) + 0.5) / np.abs(((B04 - B03) / (d)) + 0.5) * np.sqrt(np.abs((((B04 - B03) / (d))) + 0.5)));
    return CTVI
#https://www.mdpi.com/1424-8220/22/19/7440
def DRS(XTrain): #
    B04 = XTrain[:, band_dictionary['B04']]
    B012 = XTrain[:, band_dictionary['B012']]
    d=(np.power(B04,2)+np.power(B012,2))
    DRS = np.where((d <= 0), 0,np.power(d,0.5));
    return DRS
#https://www.mdpi.com/2072-4292/14/23/6105
def DRSBis(XTrain): #
    B04 = XTrain[:, band_dictionary['B04']]
    B011 = XTrain[:, band_dictionary['B011']]
    d=(np.power(B04,2)+np.power(B011,2))
    DRSBis = np.where((d <= 0), 0,np.power(d,0.5));
    return DRSBis
def DSWI3(XTrain): # B03 / B04
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    DSWI3 = np.where((B04 == 0), 0,B03 / B04);
    return DSWI3
#Disease Stress water index
#https://www.sciencedirect.com/science/article/pii/S0034425720306131?via%3Dihub#f0010 ##corretto rispetto alla prima verisone
def DSWI(XTrain): # (B08+B03)/(B11+B04) https://zslpublications.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Frse2.93&file=rse293-sup-0001-AppendixS1.docx
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    B11 = XTrain[:, band_dictionary['B011']]
    d=(B11+B04)
    DSWI = np.where((d == 0), 0,(B08+B03)/d)
    return DSWI
#https://www.mdpi.com/1424-8220/22/19/7440
def DWSI(XTrain): # (B08+B03)/(B11+B4)
    B03 = XTrain[:, band_dictionary['B03']]
    B08 = XTrain[:, band_dictionary['B08']]
    B04 = XTrain[:, band_dictionary['B04']]
    B11 = XTrain[:, band_dictionary['B011']]
    d=(B04+B11)
    DWSI = np.where((d == 0), 0,(B08+B03)/d)
    return DWSI
#https://www.mdpi.com/2072-4292/14/23/6105
def DWSIBis(XTrain): # (B08-B03)/(B11+B4)
    B03 = XTrain[:, band_dictionary['B03']]
    B08 = XTrain[:, band_dictionary['B08']]
    B04 = XTrain[:, band_dictionary['B04']]
    B11 = XTrain[:, band_dictionary['B011']]
    d=(B04+B11)
    DWSIBis = np.where((d == 0), 0,(B08-B03)/d)
    return DWSIBis
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
def EVI(XTrain): #EVI (Sentinel 2) = 2.5 * ((B8 – B4) / (B8 + 6 * B4 – 7.5 * B2 + 1))
    B02 = XTrain[:, band_dictionary['B02']]
    B04 = XTrain[:, band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=(B08+6*B04-7.5*B02+1)
    EVI = np.where((d == 0), 0,(2.5*((B08-B04)/d)))
    return EVI
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
def GCI(XTrain):  # = (B9 / B3) -1
    B03 = XTrain[:, band_dictionary['B03']]
    B09 = XTrain[:, band_dictionary['B09']]
    GCI = np.where((B03 == 0), 0,((B09/B03)-1))
    return GCI
#https://www.mdpi.com/1424-8220/22/19/7440
def GNDVIhyper(XTrain): #(B07-B03)/(B07+B03)
    B03 = XTrain[:, band_dictionary['B03']]
    B07 = XTrain[:, band_dictionary['B07']]
    d=(B07+B03)
    GNDVIhyper = np.where((d == 0), 0, ((B07-B03)/d))
    return GNDVIhyper
#Green Difference Vegetation Index
def GDVI(XTrain): # B08 - B03;
    B03 = XTrain[:, band_dictionary['B03']]
    B08 = XTrain[:, band_dictionary['B08']]
    GDVI=B08 - B03
    return GDVI
#https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/
def GI(XTrain):
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    GI = np.where((B04 == 0), 0, (B03/B04))
    return GI
#https://www.mdpi.com/2072-4292/14/13/3135#B55-remotesensing-14-03135
# formula da https://pro.arcgis.com/en/pro-app/latest/arcpy/image-analyst/gemi.htm
#formula da https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/
def GEMI(XTrain): # ((2.0 * (Math.pow(B08, 2.0) - Math.pow(B04, 2.0)) + 1.5 * B08 + 0.5 * B04) / (B08 + B04 + 0.5) * (1.0 - 0.25 * (2.0 * (Math.pow(B08, 2.0) - Math.pow(B04, 2.0)) + 1.5 * B08 + 0.5 * B04) / (B08 + B04 + 0.5)) - (B04 - 0.125) / (1.0 - B04))
    B08 = XTrain[:, band_dictionary['B08']]
    B04 = XTrain[:, band_dictionary['B04']]
    d1 = (B08 + B04 + 0.5)
    d2 = (B08 + B04 + 0.5)
    d3 = (1.0 - B04)
    cond=(d1 == 0) | (d2==0) | (d3==0)
    GEMI =np.where(cond==True, 0, ((2.0 * (np.power(B08, 2.0) - np.power(B04, 2.0)) + 1.5 * B08 + 0.5 * B04)
           /
           (d1)
           *
           (1.0 - 0.25 * (2.0 * (np.power(B08, 2.0) - np.power(B04, 2.0)) + 1.5 * B08 + 0.5 * B04) / (d2))
           -
           (B04 - 0.125) / (d3)))

    return GEMI
#Green leaf index
#https://www.mdpi.com/1424-8220/22/19/7440
#https://www.sciencedirect.com/science/article/pii/S0034425720306131?via%3Dihub#f0010
def GLI(XTrain): #(2.0 * B03 - B04 - B02) / (2.0 * B03 + B04 + B02);
    B02 = XTrain[:, band_dictionary['B02']]
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    d=2.0 * B03 + B04 + B02
    GLI=np.where((d == 0), 0,(2.0 * B03 - B04 - B02)/d)
    return GLI
#Green Normalized Difference Vegetation Index
def GNDVI(XTrain): #(B08 - B03) / (B08 + B03);
    B03 = XTrain[:, band_dictionary['B03']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=(B08 + B03)
    GNDVI=np.where((d == 0), 0,(B08 - B03)/d)
    return GNDVI
#Leaf Chlorophyll Index
def LCI(XTrain):#(B08 - B05) / (B08 + B04);
    B04 = XTrain[:, band_dictionary['B04']]
    B05 = XTrain[:, band_dictionary['B05']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=B08+B04
    LCI = np.where((d== 0), 0, (B08 - B05) / d);
    return LCI
#https://www.mdpi.com/2072-4292/12/21/3634
def LAIGreen(XTrain):#6.753 ⋅ (B05−B04)/(B05+B04);
    B04 = XTrain[:, band_dictionary['B04']]
    B05 = XTrain[:, band_dictionary['B05']]
    d=B05+B04
    LAIGreen = np.where((d== 0), 0, 6.753*(B05 - B04) / d);
    return LAIGreen
#Leaf Water Content Index /// usata implementazione da js
def LWCI(XTrain): #Math.log(1.0 - (B08 - MIDIR)) / (-Math.log(1.0 - (B08 - MIDIR)));
    #(log⁡(1-(NIR-SWIR)))/(-log⁡(1-(NIR-SWIR)))
    MIDIR = 0.101
    B08 = XTrain[:, band_dictionary['B08']]
    d=np.where((1.0 - (B08 - MIDIR))<=0,0,(-np.log(1.0 - (B08 - MIDIR))))
    #d=(-np.log(1.0 - (B08 - MIDIR)))
    LWCI=np.where((d== 0), 0,np.log(1.0 - (B08 - MIDIR)) / d)
    return LWCI
#https://www.mdpi.com/2072-4292/14/13/3135#B55-remotesensing-14-03135
#Modified Chlorophyll Absorption in Reflectance Index
#formula da https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/
def MCARI (XTrain): # ((B05 - B04) - 0.2 * (B05 - B03)) * (B05 / B04)
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    B05 = XTrain[:, band_dictionary['B05']]
    MCARI=np.where((B04==0), 0,((B05 - B04) - 0.2 * (B05 - B03)) * (B05 / B04))
    return MCARI
#https://www.mdpi.com/2072-4292/14/13/3135#B58-remotesensing-14-03135
#formula da https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/
#formula da https://pro.arcgis.com/en/pro-app/latest/arcpy/spatial-analyst/msavi.htm
#https://www.mdpi.com/2072-4292/12/21/3634
def MSAVI (XTrain): #(2.0 * B08 + 1.0 - Math.sqrt(Math.pow((2.0 * B08 + 1.0), 2.0) - 8.0 * (B08 - B04))) / 2.0
    B04 = XTrain[:, band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=np.power((2.0 * B08 + 1.0), 2.0) - 8.0 * (B08 - B04)
    MSAVI=np.where(d<0,0,(2.0 * B08 + 1.0 - np.sqrt(d)) / 2.0)
    return MSAVI
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
def MSI(XTrain): #MSI (Sentinel 2) = B11 / B08
    B08 = XTrain[:, band_dictionary['B08']]
    B011 = XTrain[:, band_dictionary['B011']]
    MSI=np.where((B08== 0), 0,((B011/B08)-1))
    return MSI
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
def NBR(XTrain): #NBRI (Sentinel 2) = (B8 – B12) / (B8 + B12)
    B08 = XTrain[:, band_dictionary['B08']]
    B012 = XTrain[:, band_dictionary['B012']]
    d=B08+B012
    NBR=np.where((d== 0), 0,(B08-B012)/d)
    return NBR
#NDREI2—Normalized Difference Red Edge Index 2
#formula da fRemote estimation of crop and grass chlorophyll and nitrogen content using  red-edge bands on Sentinel-2 and -3 J.G.P.W. Clevers a,∗, A.A. Gitelsonb
#https://www.mdpi.com/1424-8220/22/19/7440
def NDREI2(XTrain):
    B05 = XTrain[:, band_dictionary['B05']]
    B07 = XTrain[:, band_dictionary['B07']]
    d=B05+B07
    NDREI2=np.where(d==0,0, (B07-B05)/d)
    return NDREI2
def ND790(XTrain): # (B07-B04)/(B07+B04);
    B07 = XTrain[:, band_dictionary['B07']]
    B04 = XTrain[:, band_dictionary['B04']]
    d=(B07 + B04)
    ND790 = np.where((d==0), 0, (B07 - B04)/d)
    return ND790
#https://www.mdpi.com/1424-8220/22/19/7440
def NDVI690(XTrain): # (B09-B05)/(B09+B05);
    B09 = XTrain[:, band_dictionary['B09']]
    B05 = XTrain[:, band_dictionary['B05']]
    d=(B09 + B05)
    NDVI690 = np.where((d==0), 0, (B09 - B05)/d)
    return NDVI690
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
#https://www.mdpi.com/2072-4292/12/21/3634
def NDMI(XTrain): # (B08 - B11) / (B08 + B11);
    B08 = XTrain[:, band_dictionary['B08']]
    B011 = XTrain[:, band_dictionary['B011']]
    d=(B08 + B011)
    NDMI = np.where((d==0), 0, (B08 - B011)/d)
    return NDMI
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
def NDSI(XTrain): # (B3 – B11) / (B3 + B11)
    B03 = XTrain[:, band_dictionary['B03']]
    B011 = XTrain[:, band_dictionary['B011']]
    d=(B03 + B011)
    NDSI = np.where((d==0), 0, (B03 - B011)/d)
    return NDSI
#Normalized Difference Red-Edge 2
def NDRE2(XTrain): # (B08 - B05) / (B08 + B05);
    B05 = XTrain[:, band_dictionary['B05']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=(B05 + B08)
    NDRE2 = np.where((d==0), 0, (B08 - B05)/d)
    return NDRE2
#https://www.mdpi.com/1424-8220/22/19/7440
def NDRE2Bis(XTrain): # (B07 - B05) / (B07 + B05);
    B05 = XTrain[:, band_dictionary['B05']]
    B07 = XTrain[:, band_dictionary['B07']]
    d=(B05 + B07)
    NDRE2Bis = np.where((d==0), 0, (B07 - B05)/d)
    return NDRE2Bis
#Normalized Difference Red-Edge 3 -- Manca b6
def NDRE3(XTrain): # (B08 - B06) / (B08 + B06); https://zslpublications.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Frse2.93&file=rse293-sup-0001-AppendixS1.docx
    B06 = XTrain[:, band_dictionary['B06']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=(B06 + B08)
    NDRE3 = np.where((d==0), 0, (B08 - B06)/d)
    return NDRE3
#NDRS—Normalized Distance Red and SWIR
#https://www.mdpi.com/2072-4292/14/13/3135#B58-remotesensing-14-03135
def NDRS(XTrain):
    B04 = XTrain[:, band_dictionary['B04']]
    B012 = XTrain[:, band_dictionary['B012']]
    d=B04+B012
    NDRS = np.where((d == 0), 0, (B04 - B012) / d)
    return NDRS
#Normalized Difference Vegetation Index
def NDVI(XTrain): #(B08-B04)/(B08+B04)
    B08=XTrain[:,band_dictionary['B08']]
    B04=XTrain[:,band_dictionary['B04']]
    d=(B08+B04)
    NDVI=np.where(d == 0, 0, ((B08-B04)/d))
    return NDVI
#https://www.sciencedirect.com/science/article/pii/S0303243421000428?via%3Dihub
def NDVI550_650(XTrain): #(B03-B04)/(B03+B04)
    B03=XTrain[:,band_dictionary['B03']]
    B04=XTrain[:,band_dictionary['B04']]
    d=(B03+B04)
    NDVI550_650=np.where(d == 0, 0, ((B03-B04)/d))
    return NDVI550_650
#Normalized Difference Infrared Index
def NDWI(XTrain): #(B08 - B11) / (B08 + B11)
    B08=XTrain[:,band_dictionary['B08']]
    B11=XTrain[:,band_dictionary['B011']]
    d=(B08+B11)
    NDWI=np.where(d == 0, 0, ((B08-B11)/d))
    return NDWI
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
def NDWI2(XTrain): #= (B3 – B8) / (B3 + B8)
    B03=XTrain[:,band_dictionary['B03']]
    B08=XTrain[:,band_dictionary['B08']]
    d = (B03 + B08)
    NDWI2=np.where(d == 0, 0, ((B03-B08)/d))
    return NDWI2
#Normalize Green
def NG(XTrain): #B03 / (B08 + B04 + B03);
    B03=XTrain[:,band_dictionary['B03']]
    B04=XTrain[:,band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=(B08 + B04 + B03)
    NG=np.where(d == 0, 0, (B03/d))
    return NG
#Normalized Difference Green/Red Normalized green red difference index
#https://www.mdpi.com/1424-8220/22/19/7440
def NGDRI(XTrain): #(B03 - B04) / (B03 + B04);
    B03=XTrain[:,band_dictionary['B03']]
    B04=XTrain[:,band_dictionary['B04']]
    d=(B03 + B04)
    NGDRI=np.where(d == 0, 0, ((B03 - B04)/d))
    return NGDRI
#https://www.mdpi.com/2072-4292/14/23/6105
def NMDI(XTrain): #NMDI=(NIR−(SWIR1−SWIR2))/(NIR+(SWIR1+SWIR2))
    B08 = XTrain[:, band_dictionary['B08']]
    B011 = XTrain[:, band_dictionary['B011']]
    B012 = XTrain[:, band_dictionary['B012']]
    d=(B08+B011+B012)
    NMDI = np.where(d == 0, 0, ((B08-(B011-B012)) / d))
    return NMDI
#NRVI—Normalized Ratio Vegetation Index
#https://www.mdpi.com/2072-4292/14/13/3135#B58-remotesensing-14-03135
def NRVI(XTrain): #(B08 - B04) / (B08 + B04)
    B08=XTrain[:,band_dictionary['B08']]
    B04=XTrain[:,band_dictionary['B04']]
    d=(B08+B04)
    NDWI=np.where(d == 0, 0, ((B08-B04)/d))
    return NDWI
#Plant biochemical index
def PBI(XTrain): #B08 / B03
    B03=XTrain[:,band_dictionary['B03']]
    B08=XTrain[:,band_dictionary['B08']]
    PBI=np.where(B03 == 0, 0, B08 / B03)
    return PBI
#Plant pigment ratio
def PGR(XTrain): #(B03-B02)/(B03+B02)
    B02=XTrain[:,band_dictionary['B02']]
    B03=XTrain[:,band_dictionary['B03']]
    d=(B03+B02)
    PGR=np.where(d == 0, 0, (B03-B02)/d)
    return PGR
#Ratio Drought Index
def RDI(XTrain): #B12 / B08;
    B08 = XTrain[:, band_dictionary['B08']]
    B12 = XTrain[:, band_dictionary['B012']]
    RDI = np.where(B08 == 0, 0, B12 / B08)
    return RDI
#https://www.mdpi.com/1424-8220/22/19/7440
def RENDVI1(XTrain): #(B05-B04)/(B05+B04)
    B04 = XTrain[:, band_dictionary['B04']]
    B05 = XTrain[:, band_dictionary['B05']]
    d=(B05+B04)
    RENDVI1 = np.where(d == 0, 0, ((B05-B04)/d))
    return RENDVI1
#https://www.mdpi.com/1424-8220/22/19/7440
def RI(XTrain): #(B05-B03)/(B05+B03)
    B03 = XTrain[:, band_dictionary['B03']]
    B05 = XTrain[:, band_dictionary['B05']]
    d=(B03+B05)
    RI = np.where(d == 0, 0, ((B05-B03)/d))
    return RI
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
def SAVI(XTrain): #•	SAVI (Sentinel 2) = (B08 – B04) / (B08 + B04 + 0.428) * (1.428)
    B04 = XTrain[:, band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=(B08 + B04 + 0.428)
    SAVI = np.where(d == 0, 0, (B08-B04)/d*(1.428))
    return SAVI
#https://giscrack.com/list-of-spectral-indices-for-sentinel-and-landsat/
def SIPI(XTrain): #•	SIPI = (NIR – Blue) / (NIR – Red)
    B02 = XTrain[:, band_dictionary['B02']]
    B04 = XTrain[:, band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    d=(B08-B04)
    SIPI = np.where(d == 0, 0, (B08-B02)/d)
    return SIPI

#https://www.mdpi.com/2072-4292/14/13/3135
#https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/id_89.js
def SLAVI(XTrain): #•	B08 / (B04 + B12);
    B08 = XTrain[:, band_dictionary['B08']]
    B04 = XTrain[:, band_dictionary['B04']]
    B012 = XTrain[:, band_dictionary['B012']]
    d=(B04+B012)
    SLAVI = np.where(d == 0, 0, B08/d)
    return SLAVI
#Simple Ratio Blue / Red
def SRBlueRed(XTrain): #B02/B04 https://zslpublications.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Frse2.93&file=rse293-sup-0001-AppendixS1.docx
    B02 = XTrain[:, band_dictionary['B02']]
    B04 = XTrain[:, band_dictionary['B04']]
    SRBlueRed = np.where(B04 == 0, 0, (B02/B04))
    return SRBlueRed
#Simple Ratio SWIR
def SRSWIR(XTrain): #B11/B12 https://zslpublications.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Frse2.93&file=rse293-sup-0001-AppendixS1.docx
    B11 = XTrain[:, band_dictionary['B011']]
    B12 = XTrain[:, band_dictionary['B012']]
    SRSWIR = np.where(B12 == 0, 0, (B11/B12))
    return SRSWIR
#https://www.sciencedirect.com/science/article/pii/S0303243421000428?via%3Dihub
#https://www.mdpi.com/2072-4292/14/23/6105
def TCW(XTrain): #cap – wetness	TCW = 0.1509B2 + 0.1973B3 + 0.3279B4 + 0.3406B8-0.7112B11-0.4572B12
    B02 = XTrain[:, band_dictionary['B02']]
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    B08 = XTrain[:, band_dictionary['B08']]
    B011 = XTrain[:, band_dictionary['B011']]
    B012 = XTrain[:, band_dictionary['B012']]
    TCW = 0.1509*B02 + 0.1973*B03 + 0.3279* B04 + 0.3406* B08 - 0.7112* B011 - 0.4572*B012
    return TCW
#https://www.mdpi.com/2072-4292/14/13/3135
#https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/id_98.js
def TVI(XTrain): #Math.sqrt((((B04 - B03) / (B04 + B03))) + 0.5);
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    d=(B04 + B03)
    r=np.where(d == 0, 0, ((((B04 - B03) / (d))) + 0.5))
    TVI = np.where(r<0, 0, np.sqrt(r))
    return TVI
#Vegetation Moisture Index
def VMI(XTrain): # ((B08 + 0.1) - (B12 + 0.02)) / ((B08 + 0.1) + (B12 + 0.02));
    B08 = XTrain[:, band_dictionary['B08']]
    B12 = XTrain[:, band_dictionary['B012']]
    d=((B08 + 0.1) + (B12 + 0.02))
    VMI=np.where(d == 0, 0,((B08 + 0.1) - (B12 + 0.02)) / (d))
    return VMI













