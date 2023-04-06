import os
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score,f1_score


def ens(res):
        
    # AllDf = pd.concat([res[0][1],res[1][1],res[2][1],res[3][1],res[4][1]],axis=0)

    df2 = pd.DataFrame(res[0][0],columns=['Actual'])
    AllDf = [x[1] for x in res]
    AllDf_arr = np.array(AllDf)
    calculatemode = mode(AllDf_arr, axis=0)[0]
    # df3 = pd.DataFrame(calculatemode,columns=['Ensemble voting'])

    # AllDf = pd.concat([AllDf,df2,df3],axis=1)
    acc = accuracy_score(np.asarray(df2),np.transpose(calculatemode))
    print(acc)
    return acc