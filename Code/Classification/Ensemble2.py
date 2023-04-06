import os
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score


path = r'Predict'
splitby = '_ex.txt'  #  _val     _train
DatasetList = os.listdir(path)

num = 0
maxAcc = 0
BestModel = ''

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

            if num == 0:
                Actual = []
                with open(FilePath, 'r') as input_file:
                    for line in input_file:
                        # try:
                        value = int(line.split()[0])
                        # except ValueError: 
                        #     value = int(line.split('[')[1].split(']')[0])
                        Actual.append(value)
                    

            df = pd.DataFrame(np.array(result),columns=[colname])

            if num == 0:
                AllDf = df
            else:
                AllDf = pd.concat([AllDf,df],axis=1)



            num = num +1

            if num == 5:
                num = 0
                print( i,colname , AllDf.shape)

                df2 = pd.DataFrame(np.array(Actual),columns=['Actual'])
                AllDf_arr = np.array(AllDf)
                calculatemode = mode(AllDf_arr, axis=1)[0]
                df3 = pd.DataFrame(np.array(calculatemode),columns=['Ensemble voting'])

                AllDf = pd.concat([AllDf,df2,df3],axis=1)
                splitbyCla = colname.split('_')[0]
                # AllDf.to_excel('all'+splitby+'_'+splitbyCla+'.xlsx',index=None)        

                acc = accuracy_score(np.asarray(df2),np.asarray(df3))
                print(acc)
                if maxAcc < acc:
                    maxAcc = acc
                    BestModel = i+' '+colname
                    AllDfFinal = AllDf 

print(BestModel)
print(maxAcc)
AllDfFinal.to_excel('all'+BestModel+'.xlsx',index=None) 
# print(AllDf.shape)

# df2 = pd.DataFrame(np.array(Actual),columns=['Actual'])
# AllDf_arr = np.array(AllDf)
# calculatemode = mode(AllDf_arr, axis=1)[0]
# df3 = pd.DataFrame(np.array(calculatemode),columns=['Ensemble voting'])

# AllDf = pd.concat([AllDf,df2,df3],axis=1)

# AllDf.to_excel('all'+splitby+'.xlsx',index=None)        


# print(accuracy_score(np.asarray(df2),np.asarray(df3)))
# # df2 = AllDf.mode(axis=1)
