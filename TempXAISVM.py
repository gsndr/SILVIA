
import numpy as np
import pandas as pd

#To elaborate SHAP on SVM

seed=42
np.random.seed(seed)
path="D:/Progetti di ricerca/2022/SWIFTT@EUSPSA/wp1/bark beetle attacks in France/finalDB/risultati per  paper/div 10000 SPECTRAL + NMDI, MCARI, NGDRI/shap/scaled svm prob=true/"
xainame='SelfSVMModel_R1_fold6_IndexesXAI_'
yname='tempY_'
independentList = ['Coastal Aerosol', 'Blue', 'Green', 'Red', 'RED1', 'RED3', 'NIR', 'Water Vapor', 'SWIR 1', 'SWIR 2']
selectedindexes = ['NMDI', 'MCARI', 'NGDRI']
independentList=independentList+selectedindexes
print("part:", 1)
xai = np.load(path + xainame + str(1) + ".npy")
YTest = np.load(path + yname + str(1) + ".npy")
print(xai.shape)
print(YTest.shape)
xai0 = xai[0, :, :]
xai1 = xai[1, :, :]
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
#print("--",rank1)
dict_from_list1 = {k: v for k, v in zip(independentList, rank1)}
sorteddict_from_list1 = sorted(dict_from_list1.items(), key=lambda x: x[1], reverse=True)
dict_from_list1 = dict(sorteddict_from_list1)
print("Rank Class 0", dict_from_list0)
print("Rank Class 1", dict_from_list1)

for i in np.arange(2,9):
    print("part:",i)
    xaiI = np.load(path+xainame+str(i)+".npy")
    YTestI=np.load(path+yname+str(i)+".npy")
    xai=np.concatenate((xai,xaiI),axis=1)
    YTest = np.concatenate((YTest, YTestI), axis=0)

    print(xaiI.shape)
    print(YTestI.shape)
    xai0 = xaiI[0, :, :]
    xai1 = xaiI[1, :, :]
    df = pd.DataFrame(YTestI, columns=list('A'))
    indexes0 = (df.index[df['A'] == 0.0].tolist())
    xai0 = xai0[indexes0, :]
    indexes1 = (df.index[df['A'] != 0.0].tolist())
    xai1 = xai1[indexes1, :]
    rank0 = np.average(xai0, axis=0)
    dict_from_list0 = {k: v for k, v in zip(independentList, rank0)}
    sorteddict_from_list0 = sorted(dict_from_list0.items(), key=lambda x: x[1], reverse=True)
    dict_from_list0 = dict(sorteddict_from_list0)

    rank1 = np.average(xai1, axis=0)
    #print("--",rank1)
    dict_from_list1 = {k: v for k, v in zip(independentList, rank1)}
    sorteddict_from_list1 = sorted(dict_from_list1.items(), key=lambda x: x[1], reverse=True)
    dict_from_list1 = dict(sorteddict_from_list1)
    print("Rank Class 0", dict_from_list0)
    print("Rank Class 1", dict_from_list1)

print(xai.shape)
print(YTest.shape)
xai0 = xai[0, :, :]
xai1 = xai[1, :, :]
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
#print("--",rank1)
dict_from_list1 = {k: v for k, v in zip(independentList, rank1)}
sorteddict_from_list1 = sorted(dict_from_list1.items(), key=lambda x: x[1], reverse=True)
dict_from_list1 = dict(sorteddict_from_list1)
print("Rank Class 0", dict_from_list0)
print("Rank Class 1", dict_from_list1)
with open("SHAPGLOBALSCALEDSVMPROB(40000).txt", "w") as f:
    np.save("SHAPSVM40000", xai)  # save shape
    np.save("YSVM40000", YTest)  # save shape
    print(xai.shape, file=f)
    print("Class 0", xai0.shape)
    print("Class 1", xai1.shape)
    print("Rank Class 0", dict_from_list0,file=f)
    print("Rank Class 1", dict_from_list1, file=f)