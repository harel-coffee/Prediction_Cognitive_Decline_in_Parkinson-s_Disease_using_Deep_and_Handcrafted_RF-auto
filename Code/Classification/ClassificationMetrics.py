import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score 
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

class ClassificationMetrics :

    def __init__(self, predictorName = '' , contin = False , DatasetName = '',best = [],foldNumber='',bestFE=[]):
        self.predictorName = predictorName
        self.contin = contin
        self.DatasetName = DatasetName
        self.best = best
        self.foldNumber = foldNumber
        self.bestFE = bestFE

    # getter method
    def get_best(self):
        return self._best
      
    # setter method
    def set_best(self, x):
        self._best = x

    # getter method
    def get_bestFE(self):
        return self._bestFE
      
    # setter method
    def set_bestFE(self, x):
        self._bestFE = x

     # getter method
    def get_DatasetName(self):
        return self._DatasetName
      
    # setter method
    def set_DatasetName(self, x):
        self._DatasetName = x


    # getter method
    def get_predictorName(self):
        return self._predictorName
      
    # setter method
    def set_predictorName(self, x):
        self._predictorName = x

    # getter method
    def get_foldNumber(self):
        return self._foldNumber
      
    # setter method
    def set_foldNumber(self, x):
        self._foldNumber = x


    # getter method
    def get_contin(self):
        return self._contin
      
    # setter method
    def set_contin(self, x):
        self._contin = x

    def Results(self,metr,res):
        Avg = mean(res)
        Std = std(res)
        out = metr + "\tmean : " + str(Avg)  +  "\tstd : "  +  str(Std) + "\n"
        # print(metr,"\tmean : ",Avg , "\tstd : " , Std)
        fol = self.get_foldNumber()
        file = open("Results"+ '_fold' +fol +".txt", "a")
        file.write(out)
        file.close()

    def Print(self , Name):
        out = Name + " : \n"
        # print("\n"+Name+" : ")
        fol = self.get_foldNumber()
        file = open("Results"+ '_fold' +fol +".txt", "a")
        file.write(out)
        file.close() 

    def PrintResults(self , MetricName,res):
        Avg = mean(res)
        Std = std(res)
        out = MetricName + "\tmean : " + str(Avg)  +  "\tstd : "  +  str(Std) + "\n"
        # print(MetricName,"\tmean : ",Avg , "\tstd : " , Std)
        fol = self.get_foldNumber()
        file = open("Results"+ '_fold' +fol +".txt", "a")
        file.write(out)
        file.close()

 

    def PrintbestParameters(self):
        AlgName = self.get_predictorName()
        FEA_FSA = self.get_DatasetName()
        best = self.get_best()
        filename = 'Best_Parameters//' + FEA_FSA +'//'+ AlgName + '.txt'
        file = open(filename, "a")  
        for be in best:    
            file.write(be+'\n')
        file.close()  
           
    def PrintbestFE(self):
        AlgName = self.get_predictorName()
        FEA_FSA = self.get_DatasetName()
        bestFE = list(self.get_bestFE())
        filename = 'Best_Parameters//' + FEA_FSA +'//FE_'+ AlgName + '.txt'
        file = open(filename, "a")  
        # for be in bestFE:    
        file.write(str(bestFE)+'\n')
        file.close()  

    def Externalscorer(self,y_external,y_pred,type):

        AlgName = self.get_predictorName()
        FEA_FSA = self.get_DatasetName()

        filename = 'External_Predict//' + FEA_FSA +'//'+ AlgName +'_'+type+'.txt'
        file = open(filename, "a")

        c = 0 
        if self.get_contin() == False:
            for i in y_external:
                out = str(y_external[c])+ "\t" +str(y_pred[c]) + "\n"
                file.write(out)
                c= c+1
            self.set_contin(True)
        else:
            self.set_contin(False)
        file.close()  

        cm = confusion_matrix(y_external, y_pred)
        ac = accuracy_score(y_external, y_pred)
        MisclassificationRate = 1 - ac

        # Specificity =  cm[0, 0] / (cm[0, 1]+cm[0, 0])
        re = recall_score(y_external, y_pred, average='weighted' , zero_division=0)
        Sensitivity = re

        pr = precision_score(y_external, y_pred, average='weighted', zero_division=0)
        fs = f1_score(y_external, y_pred, average='weighted', zero_division=0)

        return {'tn': cm[0, 0], 'fp': cm[0, 1],'fn': cm[1, 0], 'tp': cm[1, 1] 
        , 'acc': ac  , 'MisclassificationRate': MisclassificationRate , 're': re , 'Sensitivity': Sensitivity , 'pre': pr , 'f_sc': fs
        } 

    # def PrintExternalResults(self , MetricName,res):
    #     out = MetricName + "\t" + str(res)  + "\n"
    #     # print(MetricName,"\t",res)
    #     fol = self.get_foldNumber()
    #     file = open("Results"+ '_fold' +fol +".txt", "a")
    #     file.write(out)
    #     file.close()

    def confusion_matrix_scorer(self, y, y_pred,type):
        # y_pred = clf.predict(X)

        AlgName = self.get_predictorName()
        FEA_FSA = self.get_DatasetName()
        fol = self.get_foldNumber()

        filename = 'Predict//' + FEA_FSA +'//'+ AlgName + '_fold' +fol + '_' +type+'.txt'
        file = open(filename, "a")
        c = 0
        if self.get_contin() == False:
            for i in y:
                out = str(y[c])+ "\t" +str(y_pred[c]) + "\n"
                c = c + 1
                file.write(out)
            self.set_contin(True)
        else:
            self.set_contin(False)

        file.close()  



        cm = confusion_matrix(y, y_pred)
        ac = accuracy_score(y, y_pred)
        MisclassificationRate = 1 - ac

        # Specificity =  cm[0, 0] / (cm[0, 1]+cm[0, 0])
        re = recall_score(y, y_pred, average='weighted' , zero_division=0)
        Sensitivity = re

        pr = precision_score(y, y_pred, average='weighted', zero_division=0)
        fs = f1_score(y, y_pred, average='weighted', zero_division=0)

        return {'tn': cm[0, 0], 'fp': cm[0, 1],'fn': cm[1, 0], 'tp': cm[1, 1] 
        , 'acc': ac  , 'MisclassificationRate': MisclassificationRate , 're': re , 'Sensitivity': Sensitivity , 'pre': pr , 'f_sc': fs
        }    