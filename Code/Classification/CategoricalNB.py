import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score 
from sklearn.metrics import f1_score
from sklearn.naive_bayes import CategoricalNB
from ClassificationMetrics import ClassificationMetrics

def CatNB(data, labels,nof,file,filename ):
    # params = {
    #     'n_neighbors': range(1, 51),
    #     'leaf_size': range(5, 65, 5),
    # }
    # clf = GridSearchCV(GaussianNB(), params, n_jobs=-1, cv=nof)
    # clf.fit(data, labels)
    # best_parameters = clf.best_params_
    # nn = best_parameters['n_neighbors']
    # ls = best_parameters['leaf_size']

    model = CategoricalNB()
    # cvs = cross_val_score(model, data, labels, cv=nof)
    # print(cvs)
    # print(cvs.mean() , "\t" , cvs.std())


    r = ClassificationMetrics()
    r.set_predictorName("CategoricalNB")
    r.set_contin(False)
    r.set_DatasetName(filename)

    bestparameters = ''
    r.set_best(bestparameters)
    r.PrintbestParameters() 

    
    cv_results = cross_validate(model, data, labels, cv=nof, scoring=r.confusion_matrix_scorer, return_train_score = True)
    r.Print("CategoricalNB")
    r.Print("test")
    r.Results("acc" , cv_results['test_acc'])
    r.Results("MCR" , cv_results['test_MisclassificationRate'])
    r.Results("re" , cv_results['test_re'])
    r.Results("Sen" , cv_results['test_Sensitivity'])
    r.Results("pre" , cv_results['test_pre'])
    r.Results("F_S" , cv_results['test_f_sc'])
    # r.Results("Specificity" , cv_results['test_Specificity'])
    # r.Results("roc" , cv_results['test_roc'])
    # r.Results("fpr" , cv_results['test_fpr'])
    # r.Results("tpr" , cv_results['test_tpr'])
    # r.Results("auc" , cv_results['test_auc'])
    r.Print("train")
    r.Results("acc" , cv_results['train_acc'])
    r.Results("MCR" , cv_results['train_MisclassificationRate'])
    r.Results("re" , cv_results['train_re'])
    r.Results("Sen" , cv_results['train_Sensitivity'])
    r.Results("pre" , cv_results['train_pre'])
    r.Results("F_S" , cv_results['train_f_sc'])
    # r.Results("Specificity" , cv_results['train_Specificity'])
    # r.Results("roc" , cv_results['train_roc'])
    # r.Results("fpr" , cv_results['train_fpr'])
    # r.Results("tpr" , cv_results['train_tpr'])
    # r.Results("auc" , cv_results['train_auc'])
