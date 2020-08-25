from sklearn import ensemble
'''
This file loads all the models here and stores them in a dictionary.
any model which beeds to be used can be selected from here
ex: the random forest algorithm and extra trees algorithm
'''
MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
}