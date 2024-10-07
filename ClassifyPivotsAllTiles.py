# coding: utf-8

"""Disclaimer
This software is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The software
has not received final approval by the National Institute for Space Research (INPE). No warranty, expressed or implied, is made by the INPE or the
Brazil Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. The software
is provided on the condition that neither the INPE nor the Brazil Government shall be held liable for any damages resulting from the authorized or
unauthorized use of the software.

License
MIT License
Copyright (c) 2021 Rodrigues et al.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE."""

#--------------------------
#        Imports
#--------------------------
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys
import subprocess
import numpy as np
import pickle
import gc
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
# plot learning curves
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
# imbalanced classes
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
# upsampling
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
# ANOVA feature selection for numeric input and categorical output
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

# Useful links:
#https://pediaa.com/difference-between-decision-tree-and-random-forest/
#https://dataaspirant.com/handle-imbalanced-data-machine-learning/
#https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

#https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html
#https://machinelearningmastery.com/plot-a-decision-surface-for-machine-learning/
#http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/

"""imbalanced-learn

imbalanced-learn is a python package offering a number of re-sampling techniques commonly
used in datasets showing strong between-class imbalance. It is compatible with scikit-learn and is
part of scikit-learn-contrib projects.
Source: https://github.com/scikit-learn-contrib/imbalanced-learn"""


#---------------------------
# Functions:
#---------------------------
def read_pivots(file_geojson):
    pivots = gpd.read_file(file_geojson)
    pivots_polygon = pivots[pivots.geom_type == 'Polygon']
    del pivots

    print('Pivos antes do limiar NDVI:',pivots_polygon.shape)  

    return pivots_polygon


def read_pivots_ana(ano,posfix):
    source_ana = os.path.join('./shp','PivosCentraisANA'+str(ano)+'_Intersects_composite_'+posfix+'.shp')
    pivots_ana = gpd.read_file(source_ana)    
    print('Pivos ANA antes do limiar NDVI:',pivots_ana.shape)

    return pivots_ana


def detect_intersects(pivots,pivots_target):
    #https://stackoverflow.com/a/30440879
    f = lambda x:np.sum(pivots_target.intersects(x))
    
    print('Evaluating the intersection of polygons...')
    intersects_polygon = pivots['geometry'].apply(f)
    
    print('Iterating over',len(pivots[intersects_polygon > 0]),'intersects polygons...')
    data = []
    for index1, pivos_hough in pivots[intersects_polygon > 0].iterrows():
        #Verifica soh intersects entre centroids do pivos da ANA:
        tmp_intersects = pivots_target['geometry'].intersects(pivos_hough['geometry'].centroid)
        if not tmp_intersects[tmp_intersects == True].empty:
            for index, row in pivots_target[pivots_target['geometry'].intersects(pivos_hough['geometry'].centroid)].iterrows():
                dist = pivos_hough['geometry'].centroid.distance(row['geometry'].centroid)
                data.append({'id': index1, 'geometry': pivos_hough.geometry, 'id_target': row.id, 'distance': dist})
            
                
                
    gpd_tmp = gpd.GeoDataFrame(data,columns=['id','geometry','id_target','distance'])
    indices = gpd_tmp.groupby('id_target')['distance'].idxmin() #https://datascience.stackexchange.com/a/30845
    pivots_success = gpd_tmp.loc[indices].drop(['id_target','distance'],axis=1)
            
    index = pivots_success.id.values #when use intersection
    pivots_miss = pivots.copy()
    pivots_miss.drop(index, inplace=True)
    pivots_success.set_index('id', inplace=True)
    

    return pivots_success, pivots_miss


def evaluate(model, test_features, test_labels):
    #https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
    """As a performance measure, accuracy is inappropriate for imbalanced classification problems.

       The main reason is that the overwhelming number of examples from the majority class (or classes) will overwhelm
       the number of examples in the minority class, meaning that even unskillful models can achieve accuracy scores of 
       90 percent, or 99 percent, depending on how severe the class imbalance happens to be."""
    predictions = model.predict(test_features)

    precision = precision_score(test_labels, predictions, average='micro') #Calculate metrics globally by counting the total true positives, false negatives and false positives.
    recall = recall_score(test_labels, predictions, average='micro')
    accuracy = accuracy_score(test_labels, predictions)
    score = f1_score(test_labels, predictions, average='micro')

    print('Model Performance')
    print('Accuracy: {:0.2f}%.'.format(accuracy))
    print('Precision: %.3f' % precision)
    print('Recall: %.3f' % recall)
    print('F-Measure: %.3f' % score)
    
    return accuracy

def save_load_models(op,pkl_filename):
    if op == 'save':
        # Save to file in the current working directory
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
    elif op == 'load':
        # Load from file
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model


#https://docs.w3cub.com/scikit_learn/auto_examples/model_selection/plot_learning_curve/    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    if n_jobs == None:
        n_jobs= (os.cpu_count() - 1)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def PlotFeatureImportance(X_data,y_data):
    # Plot feature importance using BRF model:
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0, n_jobs= (os.cpu_count() - 1))
    brf.fit(X_data, y_data)
        
    print('Accurancy before feature selection:',brf.score(X_data, y_data))
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, stratify=y_data, random_state=0, n_jobs= (os.cpu_count() - 1))
    
    print("BRF train accuracy: %0.3f" % brf.score(X_train, y_train))
    print("BRF test accuracy: %0.3f" % brf.score(X_test, y_test))
    
    #https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py
    importances = brf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking according BRF importance:")
    for f in range(X_data.shape[1]):
      print("%d. %s (%f)" % (f + 1, X_data.columns[indices[f]], importances[indices[f]]))    

    
    feat_importances = pd.Series(brf.feature_importances_, index=X_data.columns)
    
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    feat_importances.nlargest(20).plot(kind='barh', title='Balanced Random Forest Feature Importance\nbased on Mean Decrease in Impurity (MDI)',ax=ax)    
    ax.set_xlabel('Importances', fontweight ='bold')

    outfp = os.path.join('./png', '_'.join(name[0:4]) + '_all_BRF_FeatureImportance.png')
    plt.savefig(outfp, bbox_inches = "tight", dpi=150)
    subprocess.call(["mogrify", "-trim", outfp])
    plt.clf()
    plt.close()

    """"This problem stems from two limitations of impurity-based feature importances:
        - impurity-based importances are biased towards high cardinality features;
        - impurity-based importances are computed on training set statistics and therefore do not
          reflect the ability of feature to be useful to make predictions that generalize to the test set (when the model has enough capacity).
        Source: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py"""
    
    result = permutation_importance(brf, X_test, y_test, n_repeats=10,
                                    random_state=0, n_jobs= (os.cpu_count() - 1))
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title("Permutation Importances (test set)")

    outfp = os.path.join('./png', '_'.join(name[0:4]) + '_all_BRF_PermutationFeatureImportance.png')
    plt.savefig(outfp, bbox_inches = "tight", dpi=150)
    subprocess.call(["mogrify", "-trim", outfp])
    plt.clf()
    plt.close()
    

# Adapted from https://stackoverflow.com/a/850962
def bufcount(filename):
    f = open(filename)                  
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines

# Source: https://unix.stackexchange.com/a/590637
class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open("ClassifyPivotsAllTiles.log", "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            #this flush method is needed for python 3 compatibility.
            #this handles the flush command by doing nothing.
            #you might want to specify some extra behavior here.
            pass


   
#---------------------------
# Main:
#---------------------------
ano=2017
v_index =  'SAVI' #'NDVI'
typeOfData = 'Amplitude' #'Greenest'

sys.stdout = Logger() # Direct std out to both terminal and to log file

"""
        The coordinate reference system for all GeoJSON coordinates is a geographic coordinate reference system,
        using the World Geodetic System 1984 (WGS 84) [WGS84] datum, with longitude and latitude units of decimal
        degrees.
        ...
        Note: the use of alternative coordinate reference systems was specified in [GJ2008], but it has been
        removed from this version of the specification because the use of different coordinate reference systems
        - - especially in the manner specified in [GJ2008] - - has proven to have interoperability issues. In general,
        GeoJSON processing software is not expected to have access to coordinate reference system databases or to have
        network access to coordinate reference system transformation parameters. However, where all involved parties 
        have a prior arrangement, alternative coordinate reference systems can be used without risk of data being 
        misinterpreted. Source: RFC 7946 Butler, et al. (2016) https://tools.ietf.org/html/rfc7946#page-3"""


#---------------------------------------------------------------------------------------------------
# Label features extracted from objects identified using Circular Hough Transform and save datasets:
#---------------------------------------------------------------------------------------------------

list_file_stats = sorted(glob.glob('./stats/'+typeOfData+'SR_'+v_index+'_composite_'+str(ano)+'_*_ConvexHullStatistics.csv'))
print('Nr. of files Stats available:',len(list_file_stats))
for file_stats in list_file_stats:
    name_geojson = os.path.join('./geojson', os.path.basename(file_stats.split('_Convex')[0]))+'.geojson'
    
    #Verify there is a dataframe validated with ANA mapped information:
    outfp = os.path.join('./pickle',os.path.basename(name_geojson).split('.')[0] + '_df_validated.pkl')

    if not os.path.exists(outfp):
        print('Processing file:', os.path.basename(file_stats))  
        
        #Read spatial information of circles:
        pivots = read_pivots(name_geojson)
        pivots.reset_index(drop=True,inplace=True)
        pivots = pivots.drop('id', axis=1)
        pivots.index.names = ['id']
        nr_total_pivots = pivots.shape[0]

        print('Memory usage (KB) of pivots geodataframe:',sys.getsizeof(pivots)/1024)

        

        #Read statistics information of circles:
        data_types = {'i': np.int64,'x': np.int64,'y': np.int64,'radius': np.int64,'DifPointsPercent': np.float64,
                      'PixelsWithin': np.int64,'Mean_ndvi': np.float64,'Std_ndvi': np.float64,'Mean_savi': np.float64,
                      'Std_savi': np.float64,'Mean_amplitude': np.float64,'Std_amplitude': np.float64,'Mode_lulc': np.int64}
        
        df_pivots = pd.read_csv(file_stats, error_bad_lines=False) # About bad lines - https://stackoverflow.com/a/35212740
        
        print('Memory usage (KB) of stats dataframe:', sys.getsizeof(df_pivots)/1024)
        print(df_pivots[df_pivots.isin([np.nan, np.inf, -np.inf]).any(1)])

        df_pivots = df_pivots.dropna()
        df_pivots = df_pivots.astype(data_types)
                
        nr_lines_correct = df_pivots.shape[0]
        nr_lines_total = bufcount(file_stats) - 1 #file have a head
        if nr_lines_correct != nr_lines_total:
            print('\nNr. of bad lines skipped:',nr_lines_total - nr_lines_correct)

        df_pivots = df_pivots.drop(['x','y'],axis=1)
        df_pivots = df_pivots.set_index('i')

        #Preserve only polygons with stats information:
        pivots = pivots.loc[df_pivots.index]

        dif_pivots = nr_total_pivots - pivots.shape[0]
        if dif_pivots > 0:
            print('Nr of circles removed without stats information:',dif_pivots)
        
        
        #Read spatial information of pivots mapped by ANA:
        pivots_ana = read_pivots_ana(ano,'_'.join(os.path.basename(name_geojson).split('.')[0].split('_')[4:6]))

        #Label statistics information using spatial intersects operation between pivots mapped by ANA and identified by Hough Transform:
        pivots_success, pivots_miss = detect_intersects(pivots,pivots_ana)
        print('Aqui pivot success:',pivots_success.shape[0])

        df_final = df_pivots.copy()
        df_final['Pivot'] = 'No'
        df_final.loc[pivots_success.index,('Pivot')] = 'Yes'          

        #summary information:
        class_summary = df_final.Pivot.value_counts()        
        print('Final number of pivots after validation:\n',class_summary)
        if class_summary.No != df_pivots.shape[0]:
            if class_summary.Yes < pivots_success.shape[0]:
                print('There area some circles overlaping the same pivot mapped by ANA!')
            
        #Cleaning variables
        del df_pivots,pivots,pivots_success, pivots_miss
        
        # Save dataframe to pickle:
        df_final.to_pickle(outfp)
        del df_final



#----------------------------------
# Read all datasets labed:
#---------------------------------- 

list_file_df_validated = sorted(glob.glob('./pickle/'+typeOfData+'SR_'+v_index+'_composite_'+str(ano)+'_*_df_validated.pkl'))
print('Nr. of files dataframe validated with ANA mapped information available:',len(list_file_df_validated))
if len(list_file_df_validated) > 0:
    df_final = pd.DataFrame()

for file_df_validated in list_file_df_validated:
      print(file_df_validated)
      path = int(os.path.basename(file_df_validated).split('_')[4])
      row = int(os.path.basename(file_df_validated).split('_')[5])
      
      df_tmp = pd.read_pickle(file_df_validated)
      df_tmp.reset_index(inplace=True) #convert index i in column
      df_tmp['path'] = path #include path e row to maintain original identification
      df_tmp['row'] = row
      df_final = df_final.append(df_tmp, ignore_index=True)
      


print('Memory usage (KB) all stats dataframes:', sys.getsizeof(df_final)/1024)
del df_tmp
name = os.path.basename(file_df_validated).split('_')


#---------------------------
# View dataset:
#---------------------------
# The classes are heavily skewed we need to solve this issue later.
teste = df_final['Pivot'].value_counts(normalize=True,ascending=True)
for i in range(len(teste)):
    print('Pivots('+teste.index[i]+')', round(teste.values[i]*100,2), '% of the dataset')        

outfp = os.path.join('./png','_'.join(name[0:4]) + '_' + 'all_df_ClassDistributions.png')
if os.path.exists(outfp) == False:
    colors = ["#0101DF", "#DF0101"]
    chart = sns.countplot(x='Pivot', data=df_final, order=['Yes','No'], palette=colors)

    show_count=True
    if show_count is True:
        for i,p in enumerate(chart.patches):
            height = p.get_height()
            chart.text(p.get_x() + p.get_width() / 2.,
                               height + 30,
                               str(round(teste.values[i]*100,2))+'%',
                               ha="center")
    plt.title('Class Distributions', fontsize=14)
    # Save the figure as png file with resolution of 150 dpi
    plt.savefig(outfp, dpi=150)
    plt.clf()
    plt.close()

    # to crop png https://askubuntu.com/questions/351767/how-to-crop-borders-white-spaces-from-image
    subprocess.call(["mogrify", "-trim", outfp])


#-------------------------------------------------------------------
# ANOVA feature selection for numeric input and categorical output
# Source: https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data
# 
# As F-test (f_classif) captures only linear dependency, it rates x_1 as the most discriminative feature. On the other hand, mutual information (mutual_info_classif) can capture
# any kind of dependency between variables and it rates x_2 as the most discriminative feature, which probably agrees better with our intuitive perception for this example. 
# Both methods correctly marks x_3 as irrelevant.
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py
#-------------------------------------------------------------------
outfp_pkl = os.path.join('./pickle','_'.join(name[0:4]) + 'all_testeANOVA_FeatureSelected.pkl')
if os.path.exists(outfp_pkl) == False:
    X_train = df_final.drop(['i','Pivot','path','row'],axis=1)
    y_train = df_final.Pivot
    print('Shape of X_train=>',X_train.shape)

    plot_feature_selection = False
    results=[]
    features_count=[]
    selected=[]    
    #For function mutual_info_classif all tiles:
    my_order = ['Mode_lulc','radius', 'Mean_amplitude', 'Mean_savi', 'Std_savi', 'Mean_ndvi', 'PixelsWithin',
                'DifPointsPercent', 'Std_amplitude', 'Std_ndvi']
    print('Feature selection using F-test on mutual_info_classif for feature scoring!')
    for k in range(1,(X_train.shape[1]+1)):
      # #############################################################################
      # Univariate feature selection with F-test for feature scoring
      # We use the default selection function to select the k
      # most significant features
      fs = SelectKBest(score_func=mutual_info_classif, k=k)

      # Run score function on (X, y) and get the appropriate features
      features = fs.fit(X_train, y_train)
      
      # apply feature selection
      X_train_selected= X_train.loc[:,features.get_support()]
      
      selected = X_train.columns[features.get_support()].values.tolist()
      print('Atual:',selected)

      if plot_feature_selection:
          features_count.append([selected.count(word) for word in my_order])

      #------------------------------------------------------------
      # Fit the model
      brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0, n_jobs= (os.cpu_count() - 1))
      brf.fit(X_train_selected, y_train)

      results.append(brf.score(X_train_selected, y_train))
      
      if k > 1:
          Increase = results[k-1] - results[k-2]
          print('improvement or worsening in the accuracy: {:.2f}%'.format(Increase / results[k-2] * 100.))

    if plot_feature_selection:
        df = pd.DataFrame(features_count, columns = my_order)
        df.index += 1

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(results, 'k-o')
        df.plot.bar(stacked=True,ax=ax2, colormap='gist_rainbow') #http://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/Show_colormaps
        ax.set_xlabel('K features selected',
                       fontweight ='bold')
        ax.set_ylabel('Training accurancy',
                       fontweight ='bold')
        ax.grid(True)
        ax.set_zorder(1)  # default zorder is 0 for ax and ax2
        ax.patch.set_visible(False)  # prevents ax from hiding ax2


        #place legend above plot
        plt.legend(title='Features Selected', bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3, handletextpad=0.1)

        ax2.set_yticks([])

        outfp = os.path.join('./png','_'.join(name[0:4]) + '_all_ANOVA_FeatureSelection.png')
        # Save the figure as png file with resolution of 150 dpi
        plt.savefig(outfp,bbox_inches = "tight", dpi=150)
        plt.clf()
        plt.close()

        # to crop png https://askubuntu.com/questions/351767/how-to-crop-borders-white-spaces-from-image
        subprocess.call(["mogrify", "-trim", outfp])

    print('The best number for feature selection K=',(np.argmax(results) + 1),'maximum accurancy achieved:',results[np.argmax(results)])

    #Fitting model with best selected K features
    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=(np.argmax(results) + 1))

    # Run score function on (X, y) and get the appropriate features
    features = fs.fit(X_train, y_train)

    # Save features selected to pickle:
    with open(outfp_pkl,'wb') as f:
      pickle.dump(features,f)

    # apply feature selection
    X_train_selected= X_train.loc[:,features.get_support()]

    #------------------------------------------------------------
    # Fit the model
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0, n_jobs= (os.cpu_count() - 1))
    brf.fit(X_train_selected, y_train)


    #Return the mean accuracy on the given test data and labels.
    print('Mean accuracy of training:',brf.score(X_train_selected, y_train))


#-----------------------------------
# Building and Evaluating Models:
#-----------------------------------
#https://www.analyticsvidhya.com/blog/2020/05/decision-tree-vs-random-forest-algorithm/

X = df_final.drop(['i','Pivot','path','row'],axis=1)
Y = df_final.Pivot

# Load best feature selected:
if os.path.exists(outfp_pkl):
    with open(outfp_pkl,'rb') as f:
        features = pickle.load(f)

    X = X.loc[:,features.get_support()]
    print('Shape of X_train after feature selection=>',X.shape)

X_train = X
Y_train = Y

# Creating Train and Test Sets
print('Shape of X_train=>',X_train.shape)

opc = input('Do you want apply Random upsampling? (y/n)')
if opc == 'y':
    ros = RandomOverSampler(random_state=0, n_jobs= (os.cpu_count() - 1))
    X_resampled, y_resampled = ros.fit_resample(X_train, Y_train)
    print(sorted(Counter(y_resampled).items()))
    X_train = X_resampled
    Y_train = y_resampled

brf = BalancedRandomForestClassifier(random_state=0, n_jobs= (os.cpu_count() - 1))
brf.fit(X_train, Y_train)

gc.collect()

Y_pred = brf.predict(X_train)
print('Training Set Evaluation F1-Score:',f1_score(Y_train, Y_pred, average='micro'))
print('Training Set Evaluation Accuracy:',balanced_accuracy_score(Y_train, Y_pred))

brf_pred_all_data = brf.predict(X)
print('Types of pivots in dataset:\n',df_final.Pivot.value_counts())

df_final_classified = df_final.copy()
df_final_classified['Pivot'] = brf_pred_all_data

print('Types of pivots classified:\n',df_final_classified.Pivot.value_counts())

teste = df_final_classified.Pivot == df_final.Pivot

print('Types of pivots verified:\n',df_final.Pivot[teste].value_counts())

print('Types of pivots classified NDVI > 0.5:\n',
      df_final_classified.Pivot[df_final_classified.Mean_ndvi > 0.5].value_counts())

df_temp = df_final_classified[teste]
print('Types of pivots NDVI > 0.5 verified:\n',df_temp.Pivot[df_temp.Mean_ndvi > 0.5].value_counts())


list_file_stats = sorted(glob.glob('./stats/'+typeOfData+'SR_'+v_index+'_composite_'+str(ano)+'_*_ConvexHullStatistics.csv'))
print('Nr. of files Stats available:',len(list_file_stats),'reading spatial information of this targets...')
if len(list_file_stats) > 0:
      pivots = gpd.GeoDataFrame()

for file_stats in list_file_stats:
      name_geojson = os.path.join('./geojson', os.path.basename(file_stats).split('_Convex')[0]+'.geojson')
      print(name_geojson)
      path = int(os.path.basename(file_stats).split('_')[4])
      row = int(os.path.basename(file_stats).split('_')[5])

      #Read spatial information from Circle Houth Transform:
      pivots_tmp = read_pivots(name_geojson)
      pivots_tmp.reset_index(drop=True,inplace=True)
      pivots_tmp = pivots_tmp.drop('id', axis=1)
      pivots_tmp.index.names = ['i']
      pivots_tmp.reset_index(inplace=True) #convert index i in column
      pivots_tmp['path'] = path #include path e row to maintain original identification
      pivots_tmp['row'] = row
      pivots = pivots.append(pivots_tmp, ignore_index=True)
      
      

#Remove polygons without stats information:
df_final = df_final.set_index(['i','path','row']) #Create a multiindex
pivots = pivots.set_index(['i','path','row']) #Create a multiindex
pivots = pivots.loc[df_final.index] #remove polygons without stats information

df_final_classified = df_final_classified.set_index(['i','path','row']) #Create a multiindex

#About projection warning in Geopandas serie:
"""Projected coordinate system for South America
   You first need to decide what distortion properties would you like to control for.
   That is to say, are you interested in preserving area, distance or shape? There is a decision support
   tool for selecting projections that is quite user friendly and available for free from Oregon State
   University (https://projectionwizard.org).
   Source: https://gis.stackexchange.com/a/111531
"""
#https://alexandrakindf12.wordpress.com/2013/11/18/projections-of-south-america/
#How to solve CRS projection warning of Geopandas serie - https://stackoverflow.com/a/63038899

pivots_not_filtered = pivots.loc[df_final_classified.Pivot == 'Yes']
outfp = '_'.join(name[0:4]) + '_all_NotFiltered.' + os.path.basename(name_geojson).split('.')[1]
pivots_not_filtered.to_file(os.path.join('./geojson',outfp), driver='GeoJSON')
