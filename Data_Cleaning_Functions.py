# -*- coding: utf-8 -*-
"""
    Script containing the main functions used during the Data Cleaning section
    of the project.
    
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
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
sys.path.append("tools/")
sys.path.append("data/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from scipy import stats
from textwrap import wrap


"""----------------------------------------------------------------------------------------
Function: loadData
Loads the data contained in the "final_project_dataset.pkl" file 

Arguments:
    - features_list: List of the features to be considered in the analysis
    - total_removal: Boolean that indicates whether the TOTAL values contained
    in the available data shall be removed or not
    - nanFlag: Boolean to replace NaN values by zero
    - zerosFlag: Boolean to discard rows with only NaN values for the selected
    features

Returns:
    - data: DataFrame with selected dataset
"""
def loadData(features_list,total_removal,nanFlag,zerosFlag=False):
    ### Load the dictionary containing the dataset
    with open("data/final_project_dataset.pkl", "rb") as data_file:
        data_dict = pickle.load(data_file)
        if total_removal:
            data_dict.pop('TOTAL',0)
    
    ### Converts data from the dictionary to a DataFrame
    data = featureFormat(data_dict, features_list, sort_keys = False, 
                         remove_NaN = nanFlag,remove_all_zeroes=zerosFlag)
    data = pd.DataFrame(data,index=data_dict.keys(),columns=features_list)
    print("Data succesfully loaded.")  
    ### Returns outputs    
    return data

"""----------------------------------------------------------------------------------------
Function: plot_scatter_matrix
Generates a Scatter Matrix of a given DataFrame normalized according to the
given method

Arguments:
    - data: DataFrame with selected dataset
    - method: Method to normalize the data. 
        * None --> Scatter matrix is generated with the original data
        * maxmin --> Scatter matrix is generated with normalized data
        according to the Max and Min values of each feature.
        * std --> Scatter matrix is generated with normalized data
        according to the standarized values of each feature.
    - features_list: List of the features to be considered in the analysis
"""
def plot_scatter_matrix(data,method,features_list):
    ### Scales all data according to the specified method    
    if method == "maxmin":
        scaler = MinMaxScaler()
        normdata = scaler.fit_transform(data)
    elif method == "std":
        scaler = StandardScaler()
        normdata = scaler.fit_transform(data)
    else:
        normdata = data
            
    ### Shows a scatter matrix of all features
    df = pd.DataFrame(normdata, columns = features_list)
    # Plots scatter matrix without the "poi" column as it is a boolean
    sm = pd.plotting.scatter_matrix(df, alpha=0.3,figsize=[20,24])
    ### Set figure properties to improve visualization
    fig = plt.gcf()
    ax0 = fig.get_axes()
    n = len(features_list)
    ax0 = np.array(ax0).reshape((n,n))
    labels = features_list
    labels = [ '\n'.join(wrap(l, 10)) for l in labels ]
    for ax, mode in zip(ax0[:,0], labels):
        ax.set_ylabel(mode, size=10)
    for ax, mode in zip(ax0[-1,:], labels):
        ax.set_xlabel(mode, size=10)
    # Fits the figure to the content
    fig.tight_layout()

"""----------------------------------------------------------------------------------------
Function: show_NaN
Represents the presence of NaN values using missingno package

Arguments:
    - data: DataFrame with selected dataset
    - features_list: List of the features to be considered in the analysis
    - nplots: integer to indicate whether 1 or 2 plots shall be shown
        * nplots = 1 --> Only represents missingno matrix (with sparkline)
        * nplots = 2 --> Represents missingno matrix and bar graphs
"""
def show_NaN(data,features_list,nplots):
    if nplots == 1: # Only plots NaN matrix
        msno.matrix(data, labels=True, fontsize=8,figsize=(9,10)) 
    else: # Plots both NaN matrix and bar graphs
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9, 12))
        m = msno.matrix(data, labels=True, fontsize=8,ax=ax0, sparkline=False)     
        m = msno.bar(data, labels=True, fontsize=8,ax=ax1)
        # Fits the figure to the content
        fig.tight_layout()

"""----------------------------------------------------------------------------------------
Function: discard_outliers
Removes outliers from a dataset using a certain criteria

Arguments:
    - prediction: Mandatory in case of method="Accuracy_Continuous" 
    containing the predictions obtained with the corresponding predictor
    - original: when the outliers identification does not consist on a 
    prediction scoring model but on a sample of data itself, this input is
    directly the sample of data. Otherwise, it represents the target values.
    - method: Method/Criteria to consider one point is an outlier
        * method="Zscore" --> it uses the z-score of the sample values and
        disregards all z-score values higher than "methodVal"
    - methodVal: Pass/fail criteria of the selected method
    - features_list: List of the features to be considered in the analysis
Returns:
    - cleaned_data: DataFrame without outliers
    - outliers: index of the elements consider as outliers
"""
def discard_outliers(prediction,original,method,methodVal,features_list):
    outliers = []
    if method == "Zscore": # Discard those cases with a high Z-score 
        df = pd.DataFrame(original,columns=features_list) 
        for col in df.columns:
            if col != 'poi': # Avoid labels
                # Makes a copy of the DataFrame
                df1 = df.dropna(subset=[col]).copy()
                # Replaces values by their corresponding z-score
                df1.loc[:,col] = stats.zscore(df1.loc[:,col])
                # Convert values into Absolute values
                df1.loc[:,col] = df1.loc[:,col].abs()
                # Identify those cases considered as outliers
                out = df1[df1.loc[:,col] > methodVal].index.values.tolist()
                for i in out:
                    outliers.append(i)
        outliers = list(set(outliers))
        # Remove the corresponding outliers from the original DataFrame
        for i in outliers:
            df = df.drop([i])
        cleaned_data = df
    ### Returns outputs
    return cleaned_data, outliers

    
"""----------------------------------------------------------------------------------------
Function: remove_NaN
Remove NaN values from a DataFrame using a certain criteria

Arguments:
    - df: DataFrame containing the data to be cleaned where columns are 
    the considered features
    - features_list: List of the features to be considered in the analysis
    - nanMethod: Method to replace the NaN values
        * "zeros" --> Replace all NaN values by zero
        * "mean" --> Replace all NaN values by the mean of the feature
        * "variance" --> Replace NaN values by the values of a random serie
        generated according to the mean and std. deviation of the sample 
        (thus the general variance is more or less maintained)
Returns:
    - df: DataFrame updated without NaN values
"""
def remove_NaN(df,features_list,nanMethod): 
    for col in features_list:
        if col != 'poi': # Avoid labels
            df1 = pd.DataFrame(df[col],columns=[col])
            if nanMethod == "zeros": # Replaces NaN by zeros
                df1 = df1.fillna(0)
            elif nanMethod == "mean": # Replaces NaN by the average value
                df1 = df1.fillna(df1.mean())
            elif nanMethod == "variance": # Replaces NaN by random sample
                # Generates a random sample with the characteristics of the feature
                vals = np.random.normal(df1[col].dropna().mean(),df1[col].dropna().std(),(df1[col].isnull().sum(),1))            
                val = pd.Series([x for x in vals])
                # Replace NaN values by the values of the generated sample
                count = 0
                for i in df1.index:
                    if math.isnan(df1.at[i,col]):
                        df1.at[i,col] = val[count]
                        count += 1
            df[col] = df1
    ### Returns outputs
    return df
        
    
"""----------------------------------------------------------------------------------------
Function: components_selection
Function that perfoms an initial PCA to determine the minimum number of 
components that contain most of the explained variance ratio of the dataset

Arguments:
    - df: DataFrame with selected dataset
    - features_list: List of the features to be considered in the analysis
    - scaleFlag: Boolean to scale the features
    - scaleMethod: In case of scaleFlag=True, method to be used for scaling
    the features. 
        * scaleMethod="maxmin" --> Scale using MinMaxScaler
        * scaleMethod="std" --> Scale using StandardScaler
        * scaleMethod=None --> No scaling
    - outlierFlag: Boolean to discard outliers
    - outlierMethod: In case of outlierFlag=True, method to be used for 
    discarding outliers.
        * outlierMethod="Zscore" --> Discard outliers whose z-score is higher
        than the value given in "outlierLim"
        * outlierMethod="Accuracy_Continuous" --> Discard a certain percentage
        of datapoints that result in the highest residual errors
    
    - OutlierLim: Limit value to consider a data point as an outlier
    - nanFlag: Boolean to replace NaN values by a real value
    - nanMethod: In cases nanFlag=True, method to be used to replace NaN values
        * nanMethod="zeros" --> Replace NaN values by zero
        * nanMethod="mean" --> Replace NaN values by the mean value of the 
        feature
        * nanMethod="variance" --> Replace NaN values by the values of a 
        randomly generated sample normally distributed according to the mean
        and std deviation of the feature, thus aiming to keep constant
        the variance of the feature.
    - pcaLim: Minimum explained variance ratio to determine the minimum 
    number of components
Returns:
    - normdata: DataFrame containing the cleaned data after outliers removal,
    NaN replacement and data Scaling.
    - labels: Labels of the dataset
    - features: Features of the dataset
    - pca: PCA model
    - vardf: DataFrame containing the components of the PCA
    - n_components_Sel: The number of components for which the explained
    variance ratio criteria is met
"""
def components_selection(df,features_list,scaleFlag,scaleMethod,outlierFlag,outlierMethod,outlierLim,nanFlag,nanMethod,pcaLim):
    ### Outliers identification
    if outlierFlag:
        df,outliers = discard_outliers([],df[features_list],"Zscore",outlierLim,features_list)
    ### NaN values removal
    if nanFlag:
        df = remove_NaN(df,features_list,nanMethod)
    ### Scaling Data 
    if scaleFlag:
        if scaleMethod == "maxmin": # Use MinMaxScaler
            scaler = MinMaxScaler()
            normdata = scaler.fit_transform(df)
            normdata = pd.DataFrame(normdata,index=df.index,columns=features_list)
        elif scaleMethod == "std": # Use StandardScaler
            scaler = StandardScaler()
            normdata = scaler.fit_transform(df)
            normdata = pd.DataFrame(normdata,index=df.index,columns=features_list)
        else:
            normdata = df
        
    ### Identify labels and features
    labels, features = targetFeatureSplit(normdata.values)
    ### Makes a PCA study to determine the number of components that contain
    ### most of the variance of the dataset
    # Defines PCA model
    pca = PCA()
    # Trains the model
    pca.fit(features)
    # Gets the cumulative sum of the explained variance ratio in %
    var = pca.explained_variance_ratio_.cumsum()*100
    # Identify the number of components according to criteria
    x = np.arange(1, pca.n_components_ + 1)
    n_components_Sel = x[var>=pcaLim*100][0]
    # Plots the obtained results
    fig, ax = plt.subplots(nrows=1, figsize=(9, 7))
    ax.plot(x,var, '+',linestyle='-',linewidth=2,color='red',
            label='Explained Variance Ratio')
    ax.axhline(pcaLim*100,linestyle=':', color='k',
                label=str(pcaLim)+'% explained variance')
    ax.axvline(n_components_Sel,linestyle='--', color='green',
                label='Selected n_components')
    plt.legend(loc='best')
    ax.set_ylabel('PCA explained variance ratio')  
    ax.set_xlabel('Number of components')   
    ax.set_title('PCA initial study for features selection')
    # Fits the figure to the content
    fig.tight_layout()
    # Save PCA components into a dataframe
    vardf = pd.DataFrame(pca.components_,columns=features_list[1:])
    print("The number of selected features is: {}".format(n_components_Sel))
    ### Returns outputs
    return normdata, labels, features, pca, vardf, n_components_Sel

"""----------------------------------------------------------------------------------------
Function: components_heatmap
Plots a heat map with the components of a PCA model

Arguments:
    - vardf: DataFrame containing the components of the PCA
"""
def components_heatmap(vardf): 
    # Creates figure
    fig, ax = plt.subplots(nrows=1, figsize=(12, 9))
    # Plots heat map
    ax = sns.heatmap(vardf,vmin=-1,vmax=1,center=0,annot=True,fmt=".1g",
                     cmap='RdBu',square=True)
    # Improves visualization of axes' labels
    ax.xaxis.tick_top()
    labels = [x.get_text() for x in list(ax.get_xticklabels())]
    labels = [ '\n'.join(wrap(l, 10)) for l in labels ]
    ax.set_xticklabels(labels)
    plt.xticks(rotation=45, fontsize=8) # Rotate labels
    # Fits the figure to the content
    fig.tight_layout()
    
"""----------------------------------------------------------------------------------------
Function: best_features_selection
Selection of the features giving the best scoring according to a selected
method.

Arguments:
    - selMethod: Method to select the desired features
        * selMethod="kbest" --> Selection of features using SelectKBest model
        * selMethod="percentile" --> Selection of features using 
        SelectPercentile model
    - features_list: List of the features to be considered in the analysis
    - data: DataFrame with selected dataset
    - features: Array with selected features
    - labels: array with target labels
    - nbest: If selMethod="kbest", nbest represents the number of components to
    be selected. If selMethod="percentile", nbest represents the percentile 
    of components to be selected
    - showPlot: Boolean to graphically represent the scores and the p-values
    obtained with the selection method.
    
"""
def best_features_selection(selMethod,features_list,data,features,labels,nbest,showPlot):
    ### SelectKBest method
    if selMethod == 'kbest':
        # Defines SelectKBest model 
        sel = SelectKBest(chi2, k=nbest)
        # Fit and Transform features data
        new_features = sel.fit_transform(features, labels)
    ### SelectPercetile method
    elif selMethod == 'percentile':
        # Defines SelectKBest model 
        sel = SelectPercentile(chi2, percentile=nbest)
        # Fit and Transform features data
        new_features = sel.fit_transform(features, labels)
    ### Represents obtained scores and p-values
    if showPlot == True:   
        # Creates figure
        fig, (ax0,ax1) = plt.subplots(nrows=2, figsize=(9, 9))
        # Plot scores in first axes
        ax0.bar(features_list,sel.scores_)
        ax0.set_ylabel('Scores', fontsize=12)
        # Plot scores in second axes
        ax1.bar(features_list,sel.pvalues_)
        ax1.set_ylabel('P-Values', fontsize=12)   
        # Improves visualization of axes' labels
        lab = features_list
        lab = [ '\n'.join(wrap(l, 10)) for l in lab ]
        ax0.set_xticklabels([])
        ax1.set_xticklabels(lab)
        plt.xticks(rotation=45, fontsize=8) # Rotate labels
        # Fits the figure to the content
        fig.tight_layout()        
    ### Identify the new features
    # Gets the index of the features to be discarded
    ind = []
    for i in range(len(features[0])):
        if features[0][i] not in new_features[0]:
            ind.append(i)
    # Discards undesired features
    features_selected = features_list
    for i in sorted(ind,reverse=True):
        features_selected.pop(i)
    ### Updates dataframe with the selected data
    dat = np.hstack([np.array(labels)[:,None],new_features])
    cols = ['poi']
    [cols.append(x) for x in features_selected]
    clean_data = pd.DataFrame(dat,index=data.index,columns=cols)
    ### Returns outputs
    return clean_data, new_features, features_selected
    
"""----------------------------------------------------------------------------------------
Function: save_cleaned_data
Exports the cleaned DataFrame into a pickle file in dictionary format

Arguments:
    - df: DataFrame with cleaned dataset
    - filename: Name and path of the file
"""
def save_cleaned_data(df,filename):
    with open(filename, "wb") as outfile:
        pickle.dump(df.transpose().to_dict(), outfile)
    print('Data succesfully saved.')