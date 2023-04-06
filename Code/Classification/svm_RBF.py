import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from ClassificationMetrics import ClassificationMetrics
# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer
# from skopt.plots import plot_objective, plot_histogram
from numpy import mean
# import optuna
# from skopt.space import Integer
# from skopt.space import Real
# from skopt.space import Categorical
# from skopt.utils import use_named_args
# from skopt import gp_minimize
# from skopt import BayesSearchCV

# 3
# def objective(trial):

#     # classifier_name = trial.suggest_categorical("classifier", "SVC")

#     svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
#     classifier_obj = SVC(C=svc_c, gamma="auto",kernel='rbf')

#     # score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
#     score = cross_validate(classifier_obj, x, y, cv=no ,scoring='accuracy' , return_train_score = True)
#     global accuracy_test
#     accuracy_test = mean(score['test_score'])
#     global accuracy_train
#     accuracy_train = mean(score['train_score'])
#     return accuracy_test


# 4
# global search_space
# search_space = list()
# search_space.append(Real(1e-10, 1000.0, 'log-uniform', name='C'))
# search_space.append(Real(1e-10, 1000.0, 'log-uniform', name='gamma'))

# @use_named_args(search_space)
# def evaluate_model(**params):

#     model = SVC(kernel='rbf')
#     model.set_params(**params)
#     result = cross_validate(model, x, y, cv=no ,scoring='accuracy' , return_train_score = True)
#     global accuracy_test
#     accuracy_test = mean(result['test_score'])
#     global accuracy_train
#     accuracy_train = mean(result['train_score'])
#     return 1.0 - accuracy_test

## 5
# def bayes_search_CV_init(self, estimator, search_spaces, optimizer_kwargs=None,
#                     n_iter=50, scoring=None, fit_params=None, n_jobs=1,
#                     n_points=1, iid=True, refit=True, cv=None, verbose=0,
#                     pre_dispatch='2*n_jobs', random_state=None,
#                     error_score='raise', return_train_score=False):

#     self.search_spaces = search_spaces
#     self.n_iter = n_iter
#     self.n_points = n_points
#     self.random_state = random_state
#     self.optimizer_kwargs = optimizer_kwargs
#     self._check_search_space(self.search_spaces)
#     self.fit_params = fit_params

#     super(BayesSearchCV, self).__init__(
#         estimator=estimator, scoring=scoring,
#         n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
#         pre_dispatch=pre_dispatch, error_score=error_score,
#         return_train_score=return_train_score)
        


def svmRbf(data, labels, nof,file,filename ):
    # 1
    params = {
        # 'C': [0.1, 0.5, 1, 5, 10, 50,100],
        'C': [0.1, 0.5, 1, 5, 10, 20],
        'gamma': [0.1, 0.5, 1, 3, 6, 10],
        'kernel': ['rbf']
    }
    clf = GridSearchCV(SVC(), params, n_jobs=-1, cv=nof )
    clf.fit(data, labels)
    best_parameters = clf.best_params_
    c = best_parameters['C']
    ga = best_parameters['gamma']
    model = SVC(C=c,kernel='rbf' , gamma= ga)



# 2
    # params = {
    #     'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    #     'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),    }
    # opt = BayesSearchCV(
    #     SVC(),
    #     params,
    #     n_iter=32,
    #     cv=nof
    # )
    # opt.fit(data, labels)
    # best_parameters = opt.best_params_
    # c = best_parameters['model__C']
    # ga = best_parameters['model__gamma']
    # model = SVC(C=c , gamma= ga  , kernel = 'rbf')




# 3 
    # global x 
    # global y 
    # global no
    # x = data
    # y = labels
    # no = nof
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=10)
    # print(study.best_trial)
    # print(accuracy_test)
    # print(accuracy_train)



# 4
    # global x 
    # global y 
    # global no
    # x = data
    # y = labels
    # no = nof
    # result = gp_minimize(evaluate_model, search_space)
    # print('Best Accuracy: %.3f' % (1.0 - result.fun))
    # print('Best Parameters: %s' % (result.x))


# 5

    # params = dict()
    # params['C'] = (1e-10, 1000.0, 'log-uniform')
    # params['gamma'] = (1e-10, 1000.0, 'log-uniform')
    # model = SVC(kernel='rbf')

    # BayesSearchCV.__init__ = bayes_search_CV_init

    # search = BayesSearchCV(model, search_spaces=params, n_jobs=-1, cv=5 )
    # search.fit(data, labels)
    # print(search.best_score_)
    # print(search.best_params_)








# 1
    
    r = ClassificationMetrics()
    r.set_predictorName("SVC_RBF")
    r.set_contin(False)
    r.set_DatasetName(filename)

    bestparameters = 'C= ' +str(c) + ' kernel= rbf' + 'gamma= ' +str(ga) 
    r.set_best(bestparameters)
    r.PrintbestParameters()     
    
    cv_results = cross_validate(model, data, labels, cv=nof, scoring=r.confusion_matrix_scorer, return_train_score = True)
    r.Print("SVC_RBF")
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

