import os
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score



path = r'Predict'   # External_Predict   Predict
splitby = '_ex.txt'  #  _val.txt     _train.txt    _ex.txt
DatasetList = os.listdir(path)

num = 0
for i in DatasetList:
    MachinePath = os.path.join(path,i)
    MachineList = os.listdir(MachinePath)
    for j in MachineList:
        if j.endswith(splitby):
            colname = j.replace(splitby,'')
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

print(AllDf.shape)

df2 = pd.DataFrame(np.array(Actual),columns=['Actual'])
AllDf_arr = np.array(AllDf)
calculatemode = mode(AllDf_arr, axis=1)[0]
df3 = pd.DataFrame(np.array(calculatemode),columns=['Ensemble voting'])

AllDf = pd.concat([AllDf,df2,df3],axis=1)

AllDf.to_excel('all'+splitby+'.xlsx',index=None)        


print(accuracy_score(np.asarray(df2),np.asarray(df3)))
# df2 = AllDf.mode(axis=1)
