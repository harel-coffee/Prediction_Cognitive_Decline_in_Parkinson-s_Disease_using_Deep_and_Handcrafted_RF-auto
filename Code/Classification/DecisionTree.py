import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score 
from sklearn.metrics import f1_score
from ClassificationMetrics import ClassificationMetrics


def DT(data, labels, nof,file,filename, X_External, Y_External ):
    params = {
        'max_depth': range(1, 51),
        'min_samples_split': range(2, 11)
    }
    kf = KFold(n_splits=nof,random_state=12,shuffle=True)
    clf = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=-1, cv=kf)
    clf.fit(data, labels)
    best_parameters = clf.best_params_
    md = best_parameters['max_depth']
    mss = best_parameters['min_samples_split']


    model = DecisionTreeClassifier(max_depth=md , min_samples_split=mss)
    # cvs = cross_val_score(model, data, labels, cv=nof)
    # print(cvs)
    # print(cvs.mean() , "\t" , cvs.std())
    r = ClassificationMetrics()
    r.set_predictorName("DT")
    r.set_contin(False)
    r.set_DatasetName(filename)

    bestparameters = 'max_depth= ' +str(md) + ' min_samples_split= '+str(mss)
   

    train_val = []
    val_val = []
    test_val = []
    parameter_val = []

    parameter_val.append(bestparameters)

    for train_index, test_index in kf.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        normalized_x_train = scaler.fit_transform(X_train)
        
        normalized_x_test = scaler.transform(X_test)
                
        normalized_x_external = scaler.transform(X_External)
        
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=0.95)

        # X_transformed = pca.fit_transform(normalized_x_train)
        # X_transformed_test = pca.transform(normalized_x_test)
        # X_transformed_external = pca.transform(normalized_x_external)
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_classif,mutual_info_classif

        fs = SelectKBest(score_func=f_classif, k='all')
        # learn relationship from training data
        fs.fit(normalized_x_train, y_train)
        # transform train input data
        X_transformed = fs.transform(normalized_x_train)
        # transform test input data
        X_transformed_test = fs.transform(normalized_x_test)
        
        X_transformed_external = fs.transform(normalized_x_external)
        
        
        model.fit(X_transformed,y_train)
        
        y_true1, y_pred1 = y_train, model.predict(X_transformed)
        ac = accuracy_score(y_true1, y_pred1)
        print('train',ac)

        y_true2, y_pred2 = y_test, model.predict(X_transformed_test)
        ac = accuracy_score(y_true2, y_pred2)
        print('test',ac)

        y_true3, y_pred3 = Y_External, model.predict(X_transformed_external)
        ac = accuracy_score(y_true3, y_pred3)
        print('external',ac)

        train_val.append((y_true1, y_pred1))
        val_val.append((y_true2, y_pred2))
        test_val.append((y_true3, y_pred3))

 
    fo = 1
    r = ClassificationMetrics()
    r.set_predictorName("DT")
    r.set_DatasetName(filename)
    

    r.set_foldNumber(str(fo)+'_'+str(k_fe))  

    
    for be in range (0,len(train_val)):

        r.set_foldNumber(str(fo)+'_'+str(k_fe))  
        r.Print("DT")

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



    clf = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=-1, cv=nof )
    clf.fit(data, labels)
    best_parameters = clf.best_params_
    md = best_parameters['max_depth']
    mss = best_parameters['min_samples_split']

    bestparameters = 'max_depth= ' +str(md) + ' min_samples_split= '+str(mss)

    parameter_val.append(bestparameters)
    model = DecisionTreeClassifier(max_depth=md , min_samples_split=mss)
    model.fit( data, labels)

    y_true1, y_pred1 = labels, model.predict(data)
    ac = accuracy_score(y_true1, y_pred1)
    print('train',ac)

    y_true3, y_pred3 = Y_External, model.predict(X_External)
    ac = accuracy_score(y_true3, y_pred3)
    print('external',ac)

    r.set_contin(False)

    exResult = r.Externalscorer(y_true1, y_pred1,'train')
    r.Print("External_train")
    r.PrintResults("acc" , exResult['acc'])
    r.PrintResults("MCR" , exResult['MisclassificationRate'])
    r.PrintResults("re" , exResult['re'])
    r.PrintResults("Sen" , exResult['Sensitivity'])
    r.PrintResults("pre" , exResult['pre'])
    r.PrintResults("F_S" , exResult['f_sc'])
    r.set_contin(False)

    exResult = r.Externalscorer(y_true3, y_pred3,'ex')

    r.Print("External_test")
    r.PrintResults("acc" , exResult['acc'])
    r.PrintResults("MCR" , exResult['MisclassificationRate'])
    r.PrintResults("re" , exResult['re'])
    r.PrintResults("Sen" , exResult['Sensitivity'])
    r.PrintResults("pre" , exResult['pre'])
    r.PrintResults("F_S" , exResult['f_sc'])

    r.set_best(parameter_val)
    r.PrintbestParameters()  