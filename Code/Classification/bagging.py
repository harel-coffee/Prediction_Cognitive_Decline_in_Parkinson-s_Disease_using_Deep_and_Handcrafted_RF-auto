import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score 
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ClassificationMetrics import ClassificationMetrics
from sklearn.model_selection import StratifiedKFold,KFold
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
import pandas as pd
from Ensemble_Function import ens

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram

from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# # import mrmr
def baggingClass(data, labels,nof,file,filename , X_External, Y_External):

    kf = KFold(n_splits=nof,random_state=12,shuffle=True)
    opt = BayesSearchCV(
        BaggingClassifier(),
        {
            'base_estimator': Categorical([ SVC() , DecisionTreeClassifier()]),
            'n_estimators' : (1,250),
        },
        n_iter=256,
        cv=kf,
        n_jobs=5,
        # return_train_score=True,
        # verbose = 3
        # scoring='f1'
    )

    opt.fit(data, labels)

    print("val. score: %s" % opt.best_score_)
    print("External score: %s" % opt.score(X_External, Y_External))
    print("best params: %s" % str(opt.best_params_))


    best_parameters = opt.best_params_
    bs = best_parameters['base_estimator']
    ne = best_parameters['n_estimators']


    model = BaggingClassifier(base_estimator=bs , n_estimators= ne , bootstrap=False)

    r = ClassificationMetrics()
    r.set_predictorName("BaggingClassifier")
    r.set_contin(False)
    r.set_DatasetName(filename)

    bestparameters = 'base_estimator= ' +str(bs) + ' n_estimators= '+str(ne)



    parameter_val = []

    parameter_val.append(bestparameters)
    K_F = [10,20,30,40,50]
    for k_fe in K_F:

        train_val = []
        val_val = []
        test_val = []

        train_acc = []
        val_acc= []
        test_acc = []

        for train_index, test_index in kf.split(data, labels):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]


            ### without SCALER
            normalized_x_train = X_train
            normalized_x_test = X_test
            normalized_x_external = X_External



            ### with SCALER
            # from sklearn.preprocessing import StandardScaler
            # scaler = StandardScaler()
            # normalized_x_train = scaler.fit_transform(X_train)
            # normalized_x_test = scaler.transform(X_test)
            # normalized_x_external = scaler.transform(X_External)
            




            ### without FSA
            # X_transformed = normalized_x_train
            # X_transformed_test = normalized_x_test
            # X_transformed_external = normalized_x_external


            ### with FSA  --- PCA
            # from sklearn.decomposition import PCA
            # pca = PCA(n_components=0.95)
            # X_transformed = pca.fit_transform(normalized_x_train)
            # X_transformed_test = pca.transform(normalized_x_test)
            # X_transformed_external = pca.transform(normalized_x_external)


            ### with FSA  --- ANOVA
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import f_classif,mutual_info_classif
            fs = SelectKBest(score_func=f_classif, k=k_fe)    # 'all' 
            fs.fit(normalized_x_train, y_train)
            X_transformed = fs.transform(normalized_x_train)
            X_transformed_test = fs.transform(normalized_x_test)
            X_transformed_external = fs.transform(normalized_x_external)

            cols = fs.get_support(indices=True)
            r.set_bestFE(cols)
            r.PrintbestFE()  



            ### with FSA  --- MRMR
            # normalized_x_train = pd.DataFrame(normalized_x_train)
            # y_train = pd.Series(y_train)
            # normalized_x_test = pd.DataFrame(normalized_x_test)
            # y_test = pd.Series(y_test)
            # normalized_x_external = pd.DataFrame(normalized_x_external)
            # Y_External = pd.Series(Y_External)
            # from mrmr import mrmr_classif
            # selected_features = mrmr_classif(X=normalized_x_train, y=y_train, K=k_fe)
            # X_transformed = normalized_x_train[selected_features]
            # X_transformed_test = normalized_x_test[selected_features]
            # X_transformed_external = normalized_x_external[selected_features]


            ### with FSA  --- ReliefF
            # normalized_x_train = pd.DataFrame(normalized_x_train)
            # y_train = pd.DataFrame(y_train)
            # y_train = pd.Series(y_train)
            # normalized_x_test = pd.DataFrame(normalized_x_test)
            # y_test = pd.Series(y_test)
            # normalized_x_external = pd.DataFrame(normalized_x_external)
            # Y_External = pd.Series(Y_External)
            # from skrebate import ReliefF

            # rel = ReliefF(n_features_to_select=k_fe, n_neighbors=100)
            # rel.fit(normalized_x_train,y_train)
            # selected_features = rel.top_features_[:k_fe]
            # X_transformed = normalized_x_train[selected_features]
            # X_transformed_test = normalized_x_test[selected_features]
            # X_transformed_external = normalized_x_external[selected_features]


            # clf.transform(features)

            ### with FSA  --- SURF
            # normalized_x_train = pd.DataFrame(normalized_x_train)
            # y_train = pd.Series(y_train)
            # normalized_x_test = pd.DataFrame(normalized_x_test)
            # y_test = pd.Series(y_test)
            # normalized_x_external = pd.DataFrame(normalized_x_external)
            # Y_External = pd.Series(Y_External)
            # from skrebate import ReliefF
            # selected_features = SURF(n_features_to_select=k_fe)
            # X_transformed = normalized_x_train[selected_features]
            # X_transformed_test = normalized_x_test[selected_features]
            # X_transformed_external = normalized_x_external[selected_features]


            ### with FSA  --- SURFstar
            # normalized_x_train = pd.DataFrame(normalized_x_train)
            # y_train = pd.Series(y_train)
            # normalized_x_test = pd.DataFrame(normalized_x_test)
            # y_test = pd.Series(y_test)
            # normalized_x_external = pd.DataFrame(normalized_x_external)
            # Y_External = pd.Series(Y_External)
            # from skrebate import ReliefF
            # selected_features = SURFstar(n_features_to_select=k_fe)
            # X_transformed = normalized_x_train[selected_features]
            # X_transformed_test = normalized_x_test[selected_features]
            # X_transformed_external = normalized_x_external[selected_features]




            model.fit(X_transformed,y_train)

            y_true1, y_pred1 = y_train, model.predict(X_transformed)
            ac = accuracy_score(y_true1, y_pred1)
            print('train',ac)
            train_acc.append(ac)

            y_true2, y_pred2 = y_test, model.predict(X_transformed_test)
            ac = accuracy_score(y_true2, y_pred2)
            print('test',ac)
            val_acc.append(ac)

            y_true3, y_pred3 = Y_External, model.predict(X_transformed_external)
            ac = accuracy_score(y_true3, y_pred3)
            print('external',ac)
            test_acc.append(ac)

            train_val.append((y_true1, y_pred1))
            val_val.append((y_true2, y_pred2))
            test_val.append((y_true3, y_pred3))

        print(k_fe,'*******************')
        print('train',np.mean(train_acc))
        print('val',np.mean(val_acc))
        print('test',np.mean(test_acc))
        print('*******************')

        print('ensemble')
        acctest = ens(test_val)
        print('*******************')
        MetricName = "BaggingClassifier"
        out_tr = MetricName + "\nTrain \tmean : " + str(np.mean(train_acc))  +  "\tstd : "  +  str(np.std(train_acc)) + "\n"
        out_val = out_tr+'Val ' + "\tmean : " + str(np.mean(val_acc))  +  "\tstd : "  +  str(np.mean(val_acc)) + "\n"
        out_te = out_val+'Test ' + "\tmean : " + str(np.mean(test_acc))  +  "\tstd : "  +  str(np.mean(test_acc)) + "\n"
        out = out_te+'Ensemble' + "\tmean : " + str(acctest)  + "\n"
        file = open("Results_"+MetricName+'_'+filename+ '_nof' +str(k_fe) +".txt", "a")
        file.write(out)
        file.close()



        fo = 1
        r = ClassificationMetrics()
        r.set_predictorName("BaggingClassifier")
        r.set_DatasetName(filename)
        r.set_foldNumber(str(fo)+'_'+str(k_fe))  

        for be in range (0,len(train_val)):
        
            r.set_foldNumber(str(fo)+'_'+str(k_fe))  
            r.Print("BaggingClassifier")

            fo += 1
            r.set_contin(False)
            cv_results = r.confusion_matrix_scorer(train_val[be][0], train_val[be][1],'train')

            r.Print("train")
            r.Results("acc" , cv_results['acc'])
            r.Results("MCR" , cv_results['MisclassificationRate'])
            r.Results("re" , cv_results['re'])
            r.Results("Sen" , cv_results['Sensitivity'])
            r.Results("pre" , cv_results['pre'])
            r.Results("F_S" , cv_results['f_sc'])
            r.set_contin(False)

            cv_results = r.confusion_matrix_scorer(val_val[be][0], val_val[be][1],'val')

            r.Print("validation")
            r.Results("acc" , cv_results['acc'])
            r.Results("MCR" , cv_results['MisclassificationRate'])
            r.Results("re" , cv_results['re'])
            r.Results("Sen" , cv_results['Sensitivity'])
            r.Results("pre" , cv_results['pre'])
            r.Results("F_S" , cv_results['f_sc'])
            r.set_contin(False)

            cv_results = r.confusion_matrix_scorer(test_val[be][0], test_val[be][1],'ex')

            r.Print("external_fold")
            r.Results("acc" , cv_results['acc'])
            r.Results("MCR" , cv_results['MisclassificationRate'])
            r.Results("re" , cv_results['re'])
            r.Results("Sen" , cv_results['Sensitivity'])
            r.Results("pre" , cv_results['pre'])
            r.Results("F_S" , cv_results['f_sc'])

    # clf = GridSearchCV(BaggingClassifier(), params, n_jobs=-1, cv=nof )
    # clf.fit(data, labels)
    # best_parameters = clf.best_params_
    # bs = best_parameters['base_estimator']
    # ne = best_parameters['n_estimators']


    # bestparameters = 'base_estimator= ' +str(bs) + ' n_estimators= '+str(ne)
    # parameter_val.append(bestparameters)

    # model = BaggingClassifier(base_estimator=bs , n_estimators= ne , bootstrap=False)
    # model.fit(data, labels)

    
    # y_true1, y_pred1 = labels, model.predict(data)
    # ac = accuracy_score(y_true1, y_pred1)
    # print('train',ac)

    # y_true3, y_pred3 = Y_External, model.predict(X_External)
    # ac = accuracy_score(y_true3, y_pred3)
    # print('external',ac)

    # r.set_contin(False)


    # exResult = r.Externalscorer(y_true1, y_pred1,'train')
    # r.Print("External_train")
    # r.PrintResults("acc" , exResult['acc'])
    # r.PrintResults("MCR" , exResult['MisclassificationRate'])
    # r.PrintResults("re" , exResult['re'])
    # r.PrintResults("Sen" , exResult['Sensitivity'])
    # r.PrintResults("pre" , exResult['pre'])
    # r.PrintResults("F_S" , exResult['f_sc'])
    # r.set_contin(False)

    # exResult = r.Externalscorer(y_true3, y_pred3,'ex')

    # r.Print("External_test")
    # r.PrintResults("acc" , exResult['acc'])
    # r.PrintResults("MCR" , exResult['MisclassificationRate'])
    # r.PrintResults("re" , exResult['re'])
    # r.PrintResults("Sen" , exResult['Sensitivity'])
    # r.PrintResults("pre" , exResult['pre'])
    # r.PrintResults("F_S" , exResult['f_sc'])

    r.set_best(parameter_val)
    r.PrintbestParameters() 