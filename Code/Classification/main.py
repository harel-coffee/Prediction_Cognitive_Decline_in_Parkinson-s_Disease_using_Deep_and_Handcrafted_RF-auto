import numpy as np
import pandas as pd
from AdaBoost import adaboostclass
from svm_RBF import svmRbf
from bagging import baggingClass
from BernoulliNB import BerNB
from CategoricalNB import CatNB
from ComplementNB import CompNB
from GradientBoosting import GradientBoosting
from HistGradientBoosting import histGradientBoost
from KNN import knn
from LinearDiscriminantAnalysis import LDA
from Logestic import logestic
from multiLayerPerceptron import mlp
from DecisionTree import DT
from svm_RBF import svmRbf
from GaussianProcessClassifier import GaussianProcess
from GaussianNB import GNB
from ExtraTrees import Extra
from MultinomialNB import MultiNB
from NearestCentroid import NearestCen
from PassiveAggressiveClassifier import PassiveAggressiveClass
from QuadraticDiscriminantAnalysis import QDA
from RandomForrest import RF
from RidgeClassifier import RidgeClass
from svm_linear import svmLinear
from svm_poly import svmpoly
from svm_sigmoid import svmSigmoid
from SGDClassifier import SGDClass
from XGBoost import XGB
from ClassificationMetrics import ClassificationMetrics
from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
from sklearn.utils import resample
import os

Listfiles = os.listdir('Data/')
Listfile = []


# [0:24]
Listfile = Listfiles

for i in Listfile:
    filename = i
    # path = 'Data//'+filename+'.xlsx'
    path = 'Data//'+filename
    X2 = pd.read_excel(path, sheet_name='Data', engine='openpyxl', header=0)
    X2 = X2.drop(['PATNO'], axis=1)
    X2 = X2.to_numpy()
    n_samples = X2.shape[0]
    n_features = X2.shape[1]
    Y2 = pd.read_excel(path, sheet_name='Output',
                      engine='openpyxl', header=0)
    # Y2 = Y2['MOCA_x']
    Y2 = Y2.to_numpy().flatten()
    numberOfFold = 5
    
    file = open("Results.txt", "w")
    file.close()

    # X2, y2 = shuffle(X, Y)
    X, Y = resample(X2, Y2 , replace=False, random_state=12 , stratify = Y2) 
	# X, Y = resample(X2, Y2 , replace=False) 


    # X = X2.copy()
    # Y = Y2.copy() 
    
    
    # from sklearn.preprocessing import PolynomialFeatures
    # poly = PolynomialFeatures(2)
    # X = poly.fit_transform(X)
    # print(X.shape)

    # print(X.shape)
    # from sklearn.preprocessing import PolynomialFeatures
    # poly = PolynomialFeatures(2)
    # X = poly.fit_transform(X)
    # print(X.shape)

    X, X_External, Y, Y_External = train_test_split(X, Y, test_size=0.20,random_state=12)


    dirName = "Best_Parameters"
    if not os.path.exists(dirName):
        os.mkdir(dirName) 
    dirName = "Predict"
    if not os.path.exists(dirName):
        os.mkdir(dirName) 
    dirName = "Results"
    if not os.path.exists(dirName):
        os.mkdir(dirName) 
    dirName = "External_Predict"
    if not os.path.exists(dirName):
        os.mkdir(dirName) 


    filename = filename.split('.')[0]

    dirName = "External_Predict//"+filename
    if not os.path.exists(dirName):
        os.mkdir(dirName)

    dirName = "Predict//"+filename
    if not os.path.exists(dirName):
        os.mkdir(dirName)

    dirName = "Best_Parameters//"+filename
    if not os.path.exists(dirName):
        os.mkdir(dirName) 

    e = 'Error' 
    try:
        adaboostclass(X, Y, numberOfFold, file, filename, X_External, Y_External)
    except ValueError:
        print(e)

    
    try:
        baggingClass(X, Y, numberOfFold, file, filename, X_External, Y_External)
    except ValueError:
        print(e)
    
    # # try:
    # #     BerNB(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)
    # # try:
    # #     CatNB(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)
    # # try:
    # #     CompNB(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)
    # # try:
    # #     DT(X, Y, numberOfFold, file, filename, X_External, Y_External)
    # # except ValueError:
    # #     print(e)
    try:
        Extra(X, Y, numberOfFold, file, filename, X_External, Y_External)
    except ValueError:
        print(e)
    # # try:
    # #     GNB(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)

    # # try:
    # #     GaussianProcess(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)
    try:
        GradientBoosting(X, Y, numberOfFold, file, filename, X_External, Y_External)
    except ValueError:
        print(e)
    # # try:
    # #     histGradientBoost(X, Y, numberOfFold, file, filename, X_External, Y_External)
    # # except ValueError:
    # #     print(e)
    try:
        knn(X, Y, numberOfFold, file, filename, X_External, Y_External)
    except ValueError:
        print(e)
    # # try:
    # #     LDA(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)
    # # try:
    # #     logestic(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)
    try:
        mlp(X, Y, numberOfFold, file, filename, X_External, Y_External)
    except ValueError:
        print(e)
    # # try:
    # #     MultiNB(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)
    # # try:
    # #     NearestCen(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)
    # # try:
    # #     PassiveAggressiveClass(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)


    # # try:
    # #     QDA(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)

    try:
        RF(X, Y, numberOfFold, file, filename, X_External, Y_External)
    except ValueError:
        print(e)


#     # try:
#     #     RidgeClass(X, Y, numberOfFold, file, filename)
#     # except ValueError:
#     #     print(e)


    # # try:
    # #     SGDClass(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)


    # # try:
    # #     svmLinear(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)

    # # try:
    # #     svmpoly(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)
    
    # # try:
    # #     svmRbf(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)

    # # try:
    # #     svmSigmoid(X, Y, numberOfFold, file, filename)
    # # except ValueError:
    # #     print(e)

    try:
        XGB(X, Y, numberOfFold, file, filename, X_External, Y_External )
    except ValueError:
        print(e)
    

    print(filename)       
    lFile = os.listdir(r'.')
    for ll in lFile:
        if ll.endswith('.txt') and ll != 'Results.txt':
            txtfile = 'Results//'+filename+'_'+ll
            with open(ll, 'r') as firstfile, open(txtfile, 'w') as secondfile:
                for line in firstfile:
                    secondfile.write(line)
            os.remove(ll) 
