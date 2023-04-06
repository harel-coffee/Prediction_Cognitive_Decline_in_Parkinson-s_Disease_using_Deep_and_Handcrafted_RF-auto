import os
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score


path = r'Predict'
splitby = '_ex.txt'  #  _val     _train   _ex
DatasetList = os.listdir(path)
fold1 = []
fold2 = []
fold3 = []
fold4 = []
fold5 = []
Actual1 = []
Actual2 = []
Actual3 = []
Actual4 = []
Actual5 = []
num = 0
for i in DatasetList:
    MachinePath = os.path.join(path,i)
    MachineList = os.listdir(MachinePath)
    for j in MachineList:
        if j.endswith(splitby):
            colname = j.replace(splitby,'')
            # print(colname[-1])
            FilePath = os.path.join(MachinePath,j)
            result = []
            with open(FilePath, 'r') as input_file:
                for line in input_file:
                    # try:
                    value = int(line.split()[1])
                    # except ValueError: 
                    #     value = int(line.split('[')[1].split(']')[0])
                    result.append(value)

            Actual = []
            with open(FilePath, 'r') as input_file:
                for line in input_file:
                    # try:
                    value = int(line.split()[0])
                    # except ValueError: 
                    #     value = int(line.split('[')[1].split(']')[0])
                    Actual.append(value)
                    

            df = pd.DataFrame(np.array(result),columns=[colname])
            # df2 = pd.DataFrame(np.array(Actual),columns=['Actual'])
            if num == 0:
                fold1.append(df) 
                Actual1 = Actual.copy()
            elif num == 1:
                fold2.append(df) 
                Actual2 = Actual.copy()
                # print(i,colname  , df2.shape)
            elif num == 2:
                fold3.append(df)
                Actual3 = Actual.copy()
            elif num == 3:
                fold4.append(df)
                Actual4 = Actual.copy()
            elif num == 4: 
                fold5.append(df)
                Actual5 = Actual.copy()
            



            num = num +1

            if num == 5:
                num = 0

print(len(fold1))
print(len(fold2))
print(len(fold3))
print(len(fold4))
print(len(fold5))


for i in range(0,len(fold1)):
    if i == 0:
        AllDf1 = fold1[i]
        AllDf2 = fold2[i]
        AllDf3 = fold3[i]
        AllDf4 = fold4[i]
        AllDf5 = fold5[i]
    else:
        AllDf1 = pd.concat([AllDf1,fold1[i]],axis=1)
        AllDf2 = pd.concat([AllDf2,fold2[i]],axis=1)
        AllDf3 = pd.concat([AllDf3,fold3[i]],axis=1)
        AllDf4 = pd.concat([AllDf4,fold4[i]],axis=1)
        AllDf5 = pd.concat([AllDf5,fold5[i]],axis=1)

def cal(fo,AllDf,Actual):
    print( 'Fold',str(fo) , AllDf.shape)
    df2 = pd.DataFrame(np.array(Actual),columns=['Actual'])
    AllDf_arr = np.array(AllDf)
    calculatemode = mode(AllDf_arr, axis=1)[0]
    df3 = pd.DataFrame(np.array(calculatemode),columns=['Ensemble voting'])
    # print(calculatemode)
    AllDf = pd.concat([AllDf,df2,df3],axis=1)
    splitbyCla = colname.split('_')[0]
    AllDf.to_excel('all'+splitby+'_'+splitbyCla+'.xlsx',index=None)  
    acc =accuracy_score(np.asarray(df2),np.asarray(df3))      
    print(acc)
    return acc


acc1 = cal(1,AllDf1,Actual1)
acc2 = cal(2,AllDf2,Actual2)
acc3 = cal(3,AllDf3,Actual3)
acc4 = cal(4,AllDf4,Actual4)
acc5 = cal(5,AllDf5,Actual5)

print(np.mean([acc1,acc2,acc3,acc4,acc5]))


# print(accuracy_score(np.asarray(df2),np.asarray(df3)))
# print(AllDf.shape)

# df2 = pd.DataFrame(np.array(Actual),columns=['Actual'])
# AllDf_arr = np.array(AllDf)
# calculatemode = mode(AllDf_arr, axis=1)[0]
# df3 = pd.DataFrame(np.array(calculatemode),columns=['Ensemble voting'])

# AllDf = pd.concat([AllDf,df2,df3],axis=1)

# AllDf.to_excel('all'+splitby+'.xlsx',index=None)        

# print(accuracy_score(np.asarray(df2),np.asarray(df3)))
# # df2 = AllDf.mode(axis=1)
