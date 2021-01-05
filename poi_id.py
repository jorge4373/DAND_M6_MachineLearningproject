#!/usr/bin/python
"""
    Script containing the main functions used during the Data Exploration
    section of the project.
    
    @Course: Udacity Data Analyst Nanodegree
    @Project: Module6 - Machine Learning - Identify Fraud from Enron emails
    @author: Jorge Fernández Riera
"""

"""----------------------------------------------------------------------------------------
Python Configuration
"""
import sys
import pickle
import pprint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append("tools/")
sys.path.append("data/")
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
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report
from feature_format import featureFormat, targetFeatureSplit
from Classifiers_Testing import ClassifiersTesting
from tester import dump_classifier_and_data
from textwrap import wrap


"""----------------------------------------------------------------------------------------
Function: plot_scatter_matrix
Generates a Scatter Matrix of a given DataFrame differentiating different
classes defined by a certain column of the DataFrame

Arguments:
    - df: DataFrame with selected dataset
    - classes: Column name containing the classess to be differentiate
"""
def plot_scatter_matrix(df,classes):
    ### Plots scatter Matrix
    sm = sns.pairplot(df, hue=classes, plot_kws={'alpha':0.2,'legend':'brief'})
    ### Set figure properties to improve visualization
    fig = plt.gcf() # Get figure handle
    ax0 = fig.get_axes() # Get all axes handles
    # Limit the labels size to 10 digits. If longer, line is broken
    labels = df.columns[1:]
    labels = [ '\n'.join(wrap(l, 10)) for l in labels ]
    # Set adapted labels to the corresponding axes
    countx = 0
    county = 0
    for i in range(len(ax0)):
        if ax0[i].get_ylabel():
            ax0[i].set_ylabel(labels[county], size=10)
            county += 1
        if ax0[i].get_xlabel():
            ax0[i].set_xlabel(labels[countx], size=10)
            countx += 1
    # Fits the figure to the content
    fig.tight_layout()
    
"""----------------------------------------------------------------------------------------
Function: identify_outliers
Function that performs a LOF algorithm to determine the outliers of a certain
dataset based on a given criteria. 

Arguments:
    - df: DataFrame with selected dataset
    - nn: number of neighbors for LOF method
    - cont: contamination percentage for LOF method
    - inlierRatio: percentage of data points to be kept
    - plotFlag: Boolean to indicate if plots shall be generated
    
Returns:
    - df: DataFrame updated without outliers
"""
def identify_outliers(df,nn,cont,inlierRatio,plotFlag=False):
    ### Initializes the LOF Classifier
    clf = LocalOutlierFactor(n_neighbors=nn, contamination=cont)
    ### Get the names of the features (assuming first column of the df are labels)
    features_list = df.columns[1:]
    ### For each pair of features, identify which points would be considered
    ### as an outlier according to their negative_outlier_factor
    outliers = []
    for i in range(1,len(features_list)+1):
        for j in range(1,len(features_list)+1):
            # Select the data for the corresponding pair of features
            X = np.array([df.iloc[:,j], df.iloc[:,i]]).transpose()
            # Fits and predicts which of the data points should be considered outlier
            y_pred = clf.fit_predict(X)
            # Gets the scores of the classification
            X_scores = clf.negative_outlier_factor_
            # Keeps a list of the outliers identified for each pair of features
            outList = []
            for k in X[y_pred == -1,0]:
                outList.append(np.where(np.isclose(X[:,0],k))[0][0])
            outliers.append(outList) 
    ### Counts how many times a certain data point has been considered outlier
    realOutliers = np.zeros((137,1))
    for i in outliers:
        realOutliers[i] += 1
    ### Defines the 10% of the data points that were considered outliers most of the times
    # Makes a copy of the original dataframe
    newdata = df.copy()
    # Adds a column with the number of ocurrences each row was considered outlier
    newdata['outlierFactor'] = realOutliers
    # Sorts the new column in ascending order
    newdata = newdata.sort_values(by='outlierFactor')
    # Replaces all values by one (considered outlier)
    newdata['outlierFactor'] = 1
    # Replaces the first X% of the cases by zero (considered inlier) 
    newdata.loc[0:int(newdata.shape[0]*inlierRatio),'outlierFactor'] = 0
    ### Generates a scatter matrix plot highlighting the different outliers
    if plotFlag:
        fig, axs = plt.subplots(len(features_list),len(features_list), sharey=True, figsize=(20, 24))
        count = 1
        row = 0
        for i in range(1,len(features_list)+1):
            for j in range(1,len(features_list)+1):
                # Select the data for the corresponding pair of features
                X = np.array([df.iloc[:,j], df.iloc[:,i]]).transpose()
                # Fits and predicts which of the data points should be considered outlier
                y_pred = clf.fit_predict(X)
                # Gets the scores of the classification
                X_scores = clf.negative_outlier_factor_
                # Plots all data points in black
                axs[i-1,j-1].scatter(newdata.iloc[:,j],newdata.iloc[:,i], color='k', s=3., label='Data points')
                # Highlight the outliers identified on each graph by LOF by means 
                # of a blue circle with the radius proportional to the scores
                radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
                axs[i-1,j-1].scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], s=100 * radius[y_pred == -1], edgecolors='b',
                            facecolors='none', label='Outlier scores')
                # Highlights in red the data points that were finally considered
                # as outliers
                axs[i-1,j-1].scatter(newdata.loc[newdata[newdata['outlierFactor'] == 1].index, newdata.columns[j]],
                                     newdata.loc[newdata[newdata['outlierFactor'] == 1].index, newdata.columns[i]],
                                     color='r', s=3., label='Outliers')
                # Adjusts the axes
                axs[i-1,j-1].axis('tight')
                axs[i-1,j-1].set_xlim((0, 1))
                axs[i-1,j-1].set_ylim((0, 1))
                # Set axes labels
                labels = [ '\n'.join(wrap(l, 10)) for l in features_list]
                if count % len(features_list) == 1:
                    axs[i-1,j-1].set_ylabel(labels[row], size=10)
                    row += 1
                if count > (len(features_list)*(len(features_list)-1)):
                    axs[i-1,j-1].set_xlabel(labels[count - (len(features_list)*(len(features_list)-1))-1], size=10)
                count += 1
        # Fits the figure to the content
        plt.subplots_adjust(wspace=0, hspace=0)                    
        fig.tight_layout()
        plt.show()
    ### Compares the outliers identified with the LOF method for each pair 
    ### of features with the final outliers that will be removed
    # Gets the position index of the outliers in the dataframe
    ind = []
    for i in newdata.index:
        ind.append(int(np.where(df.index == i)[0]))
    removedOutliers = ind[-(len(newdata) - int(newdata.shape[0]*inlierRatio)):]
    # Compares if each of the defined outliers were predicted by the LOF method
    # on each of the graphs
    outCheck = {'N_OK':[],'N_NOK':[]}
    for i in range(len(outliers)):
        countOK = 0
        countNOK = 0
        for j in removedOutliers:
            if j in outliers[i]:
                countOK += 1
            else:
                countNOK += 1
        outCheck['N_OK'].append(countOK)
        outCheck['N_NOK'].append(len(outliers[i])-countOK)
    dfout = pd.DataFrame(outCheck,columns=['N_OK','N_NOK']) 
    print("LOF algorithm detected an average of {:d} potential outliers on each combination of features (circled in blue).".format(int(dfout.transpose().sum().mean())))
    print("Considering all features at once, a total of {:d} outliers were finally selected as real outliers (red points), thus discarded for the analysis".format(int(len(removedOutliers))))
    print("for each pair of features, an average value of {:d} outliers predicted by the LOF method were finally removed.".format(int(dfout.mean()[0])))   
    # Remove the identified outliers from the original dataset
    ind = newdata[newdata['outlierFactor'] == 1].index
    df = df.drop(ind)
    ### Returns outputs
    return df
    
"""----------------------------------------------------------------------------------------
Function: plot_classifiers_performance
Function that generates a figure containing one subplot for each type of
classifier (representing the mean_training_score and mean_test_score results
obtained on each iteration during the GridSearch), two subplots comparing the
mean_fitting_time and mean_score_time last by each type of classifier, and a
last subplot summarizing the best scoring results obtained.

Arguments:
    - clfs: ClassifiersTesting class containing the results of the 
    GridSearch process.
"""
def plot_classifiers_performance(clfs):
    ### Gets the scoring results of the gridsearch
    dfs = clfs.score_summary(sort_by='mean_score')
    # Keeps only the interesting columns and remove the NaN values
    df1 = dfs[['estimator','mean_score','std_score']].dropna()
    ### Plots the different scores for each method (both for training and
    ### testing) and the timings of the gridsearch computation
    # Generates figure with subplots
    fig, axs = plt.subplots(7,2, figsize=(9, 20))
    # Gets the list of classifiers names
    clf = list(clfs.keys)
    clf.pop(0)
    # Initializes counters for axes reference
    row = 0
    col = 0
    # Initializes a Dictionary to store average timings and mean score with
    # best parameters configuration
    timedf = {'fitTime':[],'scoreTime':[],'best_mean':[]}
    # Loop to get the results of each method
    for m in clf:
        # Gets the means and std. scores during training
        y_train = clfs.grid_searches[m].cv_results_['mean_train_score']
        y_trainerr = clfs.grid_searches[m].cv_results_['std_train_score']
        # Gets the means and std. scores during testing
        y_test = clfs.grid_searches[m].cv_results_['mean_test_score']
        y_testerr = clfs.grid_searches[m].cv_results_['std_test_score']
        # Represents scores
        x = np.arange(1,len(y_train)+1)
        axs[row,col].fill_between(x, y_test - y_testerr, y_test + y_testerr,alpha=0.1, color='green', label='Testing std deviation')
        axs[row,col].plot(x, y_test, color='green', label='Testing mean score')
        axs[row,col].fill_between(x, y_train - y_trainerr, y_train + y_trainerr,alpha=0.1, color='red', label='Training std deviation')
        axs[row,col].plot(x, y_train, color='red', label='Training mean score')
        axs[row,col].set_title(m)
        axs[row,col].axis('tight')
        axs[row,col].set_ylabel('score')
        axs[row,col].set_xlabel('Combination of parameters')
        axs[row,col].legend(loc='best')
        # Stores average timings in the dictionary
        timedf['fitTime'].append(clfs.grid_searches[m].cv_results_['mean_fit_time'].mean())
        timedf['scoreTime'].append(clfs.grid_searches[m].cv_results_['mean_score_time'].mean())
        # Stores mean_test_score with best parameters configuration
        dd = pd.DataFrame(clfs.grid_searches[m].cv_results_)
        timedf['best_mean'].append(dd['mean_test_score'].max())
        # Increases counters for axes references
        if col == 1:
            row += 1
            col = 0
        else:
            col += 1
    # Represents the average timings during fitting in a bar plot
    timedf = pd.DataFrame(timedf, columns=timedf.keys(), index = clf)
    sm = sns.barplot(data=timedf,x=timedf.index,y='fitTime',ax=axs[row,col])
    axs[row,col].set_title('Average Fitting times')
    # Represents the average timings during fitting in a bar plot
    col += 1
    sm = sns.barplot(data=timedf,x=timedf.index,y='scoreTime',ax=axs[row,col])
    axs[row,col].set_title('Average Scoring times')
    # Represents the mean scoring with the best parameters for each method
    row += 1
    col = 0
    ax = plt.subplot2grid((7, 2), (6, 0), colspan=2)
    ax.set_title('Mean Test Scores with best parameters')
    sm = sns.scatterplot(data=timedf,x=timedf.index,y='best_mean',marker='*',s=300,color='blue',ax=ax)
    ax.grid()  
    # Fits the figure to the content 
    fig.tight_layout()
    plt.show()
  
"""----------------------------------------------------------------------------------------
Function: plot_classifiers_calibration
Function that generates a figure containing the calibration curves of the 
different types of classifiers tested (see sklearn documentation on 
https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py)

Arguments:
    - clfs: ClassifiersTesting class containing the results of the 
    GridSearch process.
    - features_train: Features dataset for training
    - labels_train: labels dataset for training
    - features_test: Features dataset for Validation testing
    - labels_test: Labels dataset for Validation testing

Returns:
    - classReport: Dictionary containing the outputs of the "classification_report"
    function (see sklearn documentation on 
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)
    
"""
def plot_classifiers_calibration(clfs, features_train, labels_train,features_test,labels_test):
    ### Defines figure layout
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ### Plots the perfectly calibrated curve
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ### Initialize dictionary to store classification information and define
    ### the labels of the classes
    classReport = {'training':{},'testing':{}}
    classes=['Non_POI','POI']
    ### plots calibration curves
    for k in clfs.grid_searches:
        # Gets best estimator for each method
        gs = clfs.grid_searches[k]
        pipe = gs.best_estimator_
        pca = pipe.steps[0][1]
        clf = pipe.steps[1][1]
        print('Best {} model: {} - {}'.format(k,pca,clf))
        # Fit and transforms features with the PCA model
        feat_train_trans = pca.fit_transform(features_train)
        feat_test_trans = pca.fit_transform(features_test)
        # Fits Classifier with the transformed features
        clf.fit(feat_train_trans,labels_train)
        # Computes the probability associated to predictions
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(feat_test_trans)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(feat_test_trans)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        # Gets metrics of the training and testing performance    
        pred_train = clf.predict(feat_train_trans)
        classReport['training'][k] = classification_report(labels_train, pred_train, output_dict=True, zero_division=0, target_names=classes)
        pred_test = clf.predict(feat_test_trans)
        classReport['testing'][k] = classification_report(labels_test, pred_test, output_dict=True, zero_division=0, target_names=classes)
        # Gets fraction of positives and mean predicted values
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(labels_test, prob_pos, n_bins=10)
        # Plots the corresponding calibration curve
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (k, ))
        # Plots histogram of probabilities
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=k,
                 histtype="step", lw=2)
    # Adds plot information
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')    
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)    
    # Fits the figure to the content 
    plt.tight_layout()
    plt.show()
    
    return classReport
  
    
"""----------------------------------------------------------------------------------------
Function: task1_select_features
Loading of the stored cleaned dataset for the selected features.

Arguments:
    - filename: File path of the pickle file containing the stored dataset 
    obtained after the "Data Cleaning" process
    - features_list: List of the names of the features selected for the analysis
"""
def task1_select_features(filename,features_list):
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.    
    print('##############################################################')
    print('Loading Data:')
    print('##############################################################')
    ### The first feature must be "poi".
    features_list = features_list # You will need to use more features
    print('Selected Features:')
    pprint.pprint(features_list)
    
    ### Load the dictionary containing the dataset
    with open(filename, "rb") as data_file:
        data_dict = pickle.load(data_file)
    # Converts data from the dictionary to a DataFrame
    data = featureFormat(data_dict, features_list, sort_keys = False, 
                         remove_NaN = False,remove_all_zeroes=False)
    data = pd.DataFrame(data,index=data_dict.keys(),columns=features_list)
    print('Data succesfully loaded.')
    # Returns outputs
    return data

"""----------------------------------------------------------------------------------------
Function: task2_remove_outliers
Identification of outliers on a given dataset by means of a Local Outlier Factor
classifier.

Arguments:
    - data: DataFrame with selected dataset
    - Nneighbors: number of neighbors for LOF method
    - contamination: contamination percentage for LOF method
    - inliers: percentage of data points to be kept
    - plotFlag: Boolean to indicate if plots shall be generated
    
Returns:
    - outdata: DataFrame without outliers 
"""
def task2_remove_outliers(data,Nneighbors,contamination,inliers,plotFlag):
    ### Task 2: Remove outliers
    print('##############################################################')
    print('Outliers identification:')
    print('##############################################################')
    outdata = identify_outliers(data,nn=Nneighbors,cont=contamination,inlierRatio=inliers,plotFlag=plotFlag)
    print('Outliers identification completed.')
    # Returns outputs
    return outdata

"""----------------------------------------------------------------------------------------
Function: task3_tune_features
Function that generates a new feature called "stock_features" that contains
the principal component of the correlated "exercised_stock_options" and 
"total_stock_value" features. After such feature is generated, the dataset can 
be scaled using a MinMaxScaler.
Finally, the function divides the dataset into labels and features, considering
as the labels the first column of the dataset.

Arguments:
    - outdata: DataFrame without outliers 
    - features_list: List of the names of the features selected for the analysis
    - corrFlag: Boolean to indicate whether the two correlated features shall
    be combined into their principal component
    - scaleFlag: Boolean to indicate whether the dataset shall be scaled using
    a MinMaxScaler
    
Returns:
    - data: Updated dataframe
    - my_dataset: DataFrame in dictionary format to be exported to a pickle file
    - features_list: Updated list of features names
    - features: list of features
    - labels: list of labels
"""
def task3_tune_features(outdata,features_list,corrFlag,scaleFlag):
    ### Task 3: Create new feature(s)
    ### Combines "exercised_stock_options" and the "total_stock_value" features
    ### into one component by means of a PCA
    if corrFlag:
        print('##############################################################')
        print('Combine correlated features:')
        print('##############################################################')
        # Defines a PCA model with 1 component
        pca = PCA(n_components=1,random_state=42)
        # Gets the correlated features
        corrdf = outdata.loc[:,["exercised_stock_options","total_stock_value"]]
        # Fits and adjusts the correlated features into the PCA model
        new_feat = pca.fit_transform(corrdf)
        # Keeps transformed component
        outdata['stock_features'] = new_feat
        # Remove the correlated/replaced features
        outdata = outdata.drop(["exercised_stock_options","total_stock_value"], axis=1)
        # Updates the list of features
        features_list = list(outdata.columns)
        print('Correlated features succesfully combined in "stock_features" component.')
        ### Plots a correlation matrix with resultant correlations
        # Gets correlation matrix
        corrdf = outdata.corr()
        # Plots a heat map with correlation values
        fig, ax = plt.subplots(figsize=(12,9))
        ax.set_title('Correlation matrix between features')
        sns.heatmap(corrdf, annot=True, fmt='.2f', ax=ax);
        # Fits the figure to the content
        plt.tight_layout()
        plt.show()
    
    ### Scales data using MinMaxScaler
    if scaleFlag:
        print('\n')
        print('###############################')
        print('Scaling data:')
        print('###############################')
        # Initializes scaler
        scaler = MinMaxScaler()
        # Fits and transform data
        normdata = scaler.fit_transform(outdata)
        # keeps data into a DataFrame
        normdata = pd.DataFrame(normdata,index=outdata.index,columns=features_list)
        print('Data scaled succesfully.')
    else:
        normdata = outdata
    # Store to my_dataset for easy export below.
    my_dataset = normdata.transpose().to_dict()

    ### Extract features and labels from dataset for local testing
    print('\n')
    print('###############################')
    print('Getting Features and Labels')
    print('###############################')
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    data = pd.DataFrame(data,index=my_dataset.keys(),columns=features_list)
    print('Features and Labels succesfully identified.')
    
    return data, features_list, features, labels, my_dataset

"""----------------------------------------------------------------------------------------
Function: task4_classifiers_search
This function divides the given dataset into two datasets (one for training
of the classifiers and one for validation testing) and executes a search of
the best estimator and configuration that provides the best scoring results
according to a defined criteria.

Arguments:
    - data: Updated dataframe
    - features_list: Updated list of features names
    - features: list of features
    - labels: list of labels
    - scoreMethod: selected scoring method to use as a pass/fail criteria
    
Returns:
    - clfs: ClassifiersTesting class containing the results of the 
    GridSearch process.
    - features_train: Features dataset for training
    - labels_train: labels dataset for training
    - features_test: Features dataset for Validation testing
    - labels_test: Labels dataset for Validation testing
"""
def task4_classifiers_search(data,features_list,features,labels,scoreMethod):
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    # Divide the dataset in training and testing subsamples
    print('\n')
    print('###############################')
    print('Dividing data set for training and validation:')
    print('###############################')
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42,
                         stratify=labels)
    print('Training and Validation data succesfully defined.')
    
    
    print('\n')
    print('###############################')
    print('Performing GridSearch over selected classifiers:')
    print('###############################')
    # Define the different classifiers to be used
    models = {'PCA':PCA(),
             'NB':GaussianNB(),
             'SVM':SVC(),
             'DT':DecisionTreeClassifier(),
             'ET':ExtraTreeClassifier(),
             'KN':KNeighborsClassifier(),
             'RN':RadiusNeighborsClassifier(),
             'AB':AdaBoostClassifier(),
             'RF':RandomForestClassifier(),
             'GB':GradientBoostingClassifier(),
             'MLPC':MLPClassifier()}
    # Define the parameters grid to be check
    params_grid = {'PCA':{'PCA__n_components':np.arange(1,features_train[0].shape[0]+1),
                          'PCA__random_state':[42]},
                    'NB':{},
                    'SVM':{'SVM__kernel':['linear','poly','rbf'],
                            'SVM__gamma':[None, 1, 10],
                            'SVM__degree':[2, 3],
                            'SVM__random_state':[42],
                            'SVM__C':[1, 10, 100]},
                    'DT':{'DT__criterion':['gini', 'entropy'],
                            'DT__min_samples_split':[1, 5, 10, 20, 50],
                            'DT__random_state':[42]},
                    'ET':{'ET__criterion':['gini', 'entropy'],
                            'ET__min_samples_split':[1, 5, 10, 20, 50],
                            'ET__random_state':[42]},
                    'KN':{'KN__n_neighbors':[5, 10, 50],
                            'KN__weights':['uniform', 'distance'],
                            'KN__algorithm':['auto', 'ball_tree', 'kd_tree']},
                    'RN':{'RN__radius':[2, 5],
                            'RN__weights':['uniform', 'distance'],
                            'RN__algorithm':['auto', 'ball_tree', 'kd_tree']},
                    'AB':{'AB__n_estimators':[10, 25, 50],
                            'AB__random_state':[42]},
                    'RF':{'RF__n_estimators':[10, 50, 100],
                            'RF__criterion':['gini', 'entropy'],
                            'RF__min_samples_split':[1, 5, 10, 20, 50],
                            'RF__random_state':[42]},
                    'GB':{'GB__n_estimators':[10, 50, 100],
                            'GB__criterion':['mse', 'mae'],
                            'GB__min_samples_split':[1, 5, 10, 20, 50],
                            'GB__random_state':[42]},
                    'MLPC':{'MLPC__max_iter':[200, 300, 500,1000,5000],
                            'MLPC__random_state':[42]}}
    # Run a GridSearch over the parameters of each classifier combined with 
    # an initial PCA step
    clfs = ClassifiersTesting(models, params_grid)
    clfs.fit(features_train, labels_train, scoring=scoreMethod, n_jobs=-1, refit=True)
    return clfs, features_train, labels_train, features_test, labels_test
    

"""----------------------------------------------------------------------------------------
Function: task4_calibration_check
Function that generates a figure containing the calibration curves of the 
different types of classifiers tested (see sklearn documentation on 
https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py)
several bar plots summarizing the precision, recall and
F1-score obtained on each case, both during training and testing of the
classifiers.

Arguments:
    - clfs: ClassifiersTesting class containing the results of the 
    GridSearch process.
    - features_train: Features dataset for training
    - labels_train: labels dataset for training
    - features_test: Features dataset for Validation testing
    - labels_test: Labels dataset for Validation testing

Returns:
    - df: DataFrame with a summary of the obtained calibration results
"""
def task4_calibration_check(clfs, features_train, labels_train, features_test, labels_test):
    ### Plots calibration results for each method
    classReport = plot_classifiers_calibration(clfs, features_train, labels_train,features_test,labels_test)
    ### Plots classification report scores
    # Generates a DataFrame with the different scores
    count = 0
    for step in ('training','testing'):
        for k in classReport[step].keys():
            if count == 0:
                df = pd.DataFrame(classReport[step][k], index=classReport[step][k]['Non_POI'].keys()).transpose()
                df = df.reset_index().rename(columns={'index':'type'})
                df['method'] = k
                df['step'] = step
            else:
                df1 = pd.DataFrame(classReport[step][k], index=classReport[step][k]['Non_POI'].keys()).transpose()
                df1 = df1.reset_index().rename(columns={'index':'type'})
                df1['method'] = k
                df1['step'] = step
                df = df.append(df1, ignore_index=True)
            count += 1
    # Generates a bar plot with Precision results
    sm = sns.catplot(x="type", y="precision",hue="method", col="step",data=df,
                    kind="bar",height=7, aspect=.7)    
    plt.suptitle('PRECISION scores')
    # Generates a bar plot with Recall results
    sm = sns.catplot(x="type", y="recall",hue="method", col="step",data=df,
                    kind="bar",height=7, aspect=.7)
    plt.suptitle('RECALL scores')
    # Generates a bar plot with F1-scores results
    sm = sns.catplot(x="type", y="f1-score",hue="method", col="step",data=df,
                    kind="bar",height=7, aspect=.7)    
    plt.suptitle('F1 scores')
    return df

"""----------------------------------------------------------------------------------------
Function: task5_select_classifier
Function that selects the best estimator classifier that provides the best
precision for "poi" identification. Alternatively, the key of one of the 
classifiers used during the gridsearch can be also given to select the best
estimator found for such classifier.

Arguments:
    - classReport: Dictionary containing the outputs of the "classification_report"
    function (see sklearn documentation on 
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)
    - clfs: ClassifiersTesting class containing the results of the 
    GridSearch process.
    - features_train: Features dataset for training
    - labels_train: labels dataset for training
    - features_test: Features dataset for Validation testing
    - labels_test: Labels dataset for Validation testing
    - selclf: key of a classifier to select the best estimator of that type
    
Returns:
    - clf: Selected classifier
    
"""
def task5_select_classifier(classReport, clfs, features_train, labels_train, features_test, labels_test, selclf):
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    print('\n')
    print('###############################')
    print('Select classifier giving best precision for "poi" identification:')
    print('###############################')
    if selclf:
        # Classifier manually defined by selclf
        best_clf_method = selclf
    else:
        # Selects from all types of classifiers the one giving best precision
        # scores to identify a "poi"
        df = classReport[classReport['step']=='testing']
        df = df[df['type']=='POI']
        best_clf_method = df.loc[df['precision'].idxmax(),'method']
    # Gets the corresponding best estimator for the selected method
    gs = clfs.grid_searches[best_clf_method]
    pipe = gs.best_estimator_
    pca = pipe.steps[0][1]
    clf = pipe.steps[1][1]
    print('Best {} model: {} - {}'.format(best_clf_method,pca,clf))
    # Fit and transforms features with the PCA model
    feat_train_trans = pca.fit_transform(features_train)
    feat_test_trans = pca.fit_transform(features_test)
    # Fits Classifier with the transformed features
    clf.fit(feat_train_trans,labels_train)
    # Predicts new results for validation sample
    pred = clf.predict(feat_test_trans)
    print('The obtained accuracy was: {}'.format(clf.score(feat_test_trans,labels_test)))
    classes=['Non_POI','POI']
    print(classification_report(labels_test, pred, target_names=classes))
    
    return clf
    
        
"""----------------------------------------------------------------------------------------
Function: task6_dump_results
Export the main information of the analysis to pickle files

Arguments:
    - clf: Selected classifier
    - my_dataset: Final dataset used for machine learning algorithms in 
    dictionary format
    features_list: Updated list of features names used for the analysis
"""
def task6_dump_results(clf, my_dataset, features_list):
    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.
    print('\n')
    print('###############################')
    print('Export obtained results:')
    print('###############################')    
    dump_classifier_and_data(clf, my_dataset, features_list)
    print('Results succesfully exported.')
    
def main():
    ### Loads cleaned data
    filename = "data/final_project_dataset_CLEANED.pkl"
    features_list = ['poi', 'bonus', 'deferred_income',
                     'exercised_stock_options', 'expenses', 'from_messages',
                     'from_poi_to_this_person', 'long_term_incentive',
                     'restricted_stock', 'salary', 'shared_receipt_with_poi',
                     'total_payments', 'total_stock_value']
    data = task1_select_features(filename,features_list)
    ### Remove Outliers
    outdata = task2_remove_outliers(data,50,0.1,0.9,True)
    ### Create new features and scale
    data, features_list, features, labels, my_dataset = task3_tune_features(outdata,features_list,True,True)
    ### Performs GridSearch over all selected classifiers and plots performance
    clfs, features_train, labels_train, features_test, labels_test = task4_classifiers_search(data,features_list,features,labels,'accuracy')
    plot_classifiers_performance(clfs)
    ### Plots calibration curves and classification report
    classReport = task4_calibration_check(clfs, features_train, labels_train, features_test, labels_test)
    ### Selects Best Estimator classifier
    clf = task5_select_classifier(classReport, clfs, features_train, labels_train, features_test, labels_test,None)
    ### Dump results
    task6_dump_results(clf, my_dataset, features_list)

if __name__ == '__main__':
    main()