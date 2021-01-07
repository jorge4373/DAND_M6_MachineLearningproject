# -*- coding: utf-8 -*-
"""
    Script containing a class for testing several types of classifiers 
    at once.
    
    
    @Course: Udacity Data Analyst Nanodegree
    @Project: Module6 - Machine Learning - Identify Fraud from Enron emails
    @author: Jorge Fernández Riera
    @Reference: http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
    Most of the code was extracted from David S.Batista blog.
"""

"""----------------------------------------------------------------------------------------
Python Configuration
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

"""----------------------------------------------------------------------------------------
Class: ClassifiersTesting
Class that allows to perform a grid search over a list of classifiers 
combined with a PCA step in a pipeline in order to determine which
combination of (pca,clf) and in which configuration provides the best scoring
results according to a certain criteria.

Functions:
    - __init__: Function that initializes the classifier and identifies 
    the different models and parameters to be considered
        * models: Dictionary where the keys are the names given to the models
        and the values are corresponding models.
        * params: Dictionary where the keys are the same names used for the 
        models and the values are dictionaries containing the parameters to be
        fine-tuned with the GridSearch algorithm. In these dictionaries, the
        keys shall be "name__parameter" being the "name" the name of the model
        and "parameter" the parameter to be varied. On the other hand, values
        are lists containing the different values of the parameter to be
        tested.
    - fit: Function that performs the GridSearch and the fitting of the 
    different models to identify the best estimator.
        * X: Features for training
        * y: labels for training
        * cv: Determines the cross-validation splitting strategy
        * n_jobs: Number of jobs to run in parallel (-1 means all processors)
        * verbose: Controls the verbosity during the search process
        * scoring: A single string or a callable to evaluate the predictions 
        on the test set
        * refit: Refit an estimator using the best found parameters on the
        whole dataset.
    - score_summary: Function that provides a DataFrame summarizing the main
    results obtained with each of the iterations of the gridsearch. Such
    DataFrame summarizes the maximum, minimum, mean and std deviation values
    of the obtained scores together with the parameters used on each iteration
        * sort_by: string indicating the column to sort the DataFrame
"""
class ClassifiersTesting:
    
    def __init__(self, models, params):
        ### Checks that inputs are properly defined
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        ### Assign variables to the class
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=10, n_jobs=-1, verbose=3, scoring=None, refit=False):
        for key in self.keys: 
            """ Changed in v2 --> Adding MinMaxScaler to pipeline
            ### Code in v1:
            if key != 'PCA': # PCA model was defined always as a first step
                try:
                    ### Define pipeline
                    try:
                        pipe = Pipeline([('PCA',self.models['PCA']),(key,self.models[key])])
            """
            ### Code in v2
            if key != 'PCA' and key != 'SC': # Scaler and PCA model was defined always as a first step
                try:
                    ### Define pipeline
                    try:
                        pipe = Pipeline([('SC',self.models['SC']),('PCA',self.models['PCA']),(key,self.models[key])])
                        """ End of changed section in v2 """
                    except:
                        raise ValueError("Pipeline could not be constructed")
                    ### Run Gridsearch over selected pipeline
                    print("\nRunning GridSearchCV for %s." % pipe)
                    # Gets parameters for the classifier
                    params_grid = self.params[key]
                    # Adds parameters for the PCA model
                    for k in self.params['PCA']:
                        params_grid[k] = self.params['PCA'][k]
                    # Defines GridSearch model
                    gs = GridSearchCV(pipe, params_grid, cv=cv, n_jobs=n_jobs,
                                      verbose=verbose, scoring=scoring, refit=refit,
                                      return_train_score=True)
                    # Executes GridSearch over selected pipeline and parameters
                    gs.fit(X,y) 
                    print("\tBest parameter (CV score=%0.3f):" % gs.best_score_)
                    print("\t",gs.best_params_)
                    ### Stores results of the search in the class
                    self.grid_searches[key] = gs    
                except:
                    raise ValueError("Error during Classifiers Testing - ", pipe)

    def score_summary(self, sort_by='mean_score'):
        ### Subfunction that returns the main statistics of the obtained
        ### scores.
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})
        ### Obtain the results obtained during the GridSearch on each iteration
        rows = []
        for k in self.grid_searches:
            # Gets the parameters of the iteration
            params = self.grid_searches[k].cv_results_['params']
            # Gets the scores obtained on each split of the CV
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))
            # Calculate the main statistics of the results of the iteration
            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))
        ### Add configuration information to the DataFrame
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        ### Returns outputs  
        return df[columns]