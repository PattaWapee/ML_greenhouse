import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier



###################################################
############Data & feature processing #############
###################################################

def scaling_features(df, col_list ,scale_method='standard'):
    """
    Description
        -----------
        sclae features
        Parameters
        ----------
            df : Pandas data frame of features
            col_list : list of column names for feature scaling
            scale_method : (default standard) scaler to use
        Returns
        -------
            df_scaled : scaled features
        
    """ 
    if scale_method == 'standard':
        scaler = preprocessing.StandardScaler()
    if scale_method == 'minmax':
        scaler = preprocessing.MinMaxScaler()

    x_scaled = scaler.fit_transform(df[col_list].values)

    for i, col in enumerate(col_list):
        df[col] = x_scaled[:,i]
    return df

def time_features(df, time_col = 'index'):
    """
    Description
        -----------
        Conversion of timestamp to other features: 
        day_of_month, month,day_of_week,hour
        Parameters
        ----------
            df : Pandas data frame of 
        Returns
        -------
            df : Pandas data frame of features 
    """
    if time_col == 'index':
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['weekday'] = df.index.dayofweek
        df['month']= df.index.month  
        
    else:
        df['hour']= np.uint8(df[time_col].dt.hour)
        df['day']= np.uint8(df[time_col].dt.day)   #day of month
        df['weekday']= np.uint8(df[time_col].dt.weekday)   #day of week
        df['month']= np.uint8(df[time_col].dt.month)  
        df.drop([time_col],axis=1,inplace=True)
    return df


def lr_predict_missing_value_(df, col_list):
    pass


################ linear regression prediction #################

def lr_predict(x_df, y_df, test_size=0.2):
    """
        Description
        -----------
        Predict Y from linear regression model
        Parameters
        ----------
            x_df : Pandas data frame of features 
            y_df : Pandas data frame of target
            test_size : (float, default 0.2) proportion of test data
        Returns
        -------
            Y_predict : prediction of target
    """
    X_train, X_test, Y_train, Y_test = train_test_split(x_df, y_df, test_size=test_size)

    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    Y_predict = lr.predict(X_test)

    mae, mse, r2 = lr_eval(Y_test, Y_predict)
    print('Mean absolute error:', mae, 'Mean square error:', mse, 'R2:', r2)
    return lr, Y_predict, Y_test

def lr_eval(Y_test, Y_predict):
    """
        Description
        -----------
        Evaluate linear regression model
        Parameters
        ----------
            Y_test : (Pandas dataframe, Pandas series,numpy array) target test data
            Y_predict : (Pandas dataframe, Pandas series,numpy array) target prediction
        Returns
        -------
            mae : Mean absolute error
            mse : Mean square error
            r2 : R2 score
    """
    mae = metrics.mean_absolute_error(Y_test, Y_predict)
    mse = metrics.mean_squared_error(Y_test, Y_predict)
    r2 = metrics.r2_score(Y_test, Y_predict)
    print('Mean absolute error:', mae, 'Mean square error:', mse, 'R2:', r2)
    return mae, mse, r2

def lr_plot(Y_test, Y_predict):
    """
        Description
        -----------
        Plot linear regression model
        Parameters
        ----------
        Returns
        -------
            Showing plot
    """
    fig, ax = plt.subplots()
    ax.scatter(Y_predict, Y_test, edgecolors=(0, 0, 1))
    ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()

    return ax

######################################################
################ Classification model ################
######################################################

def GetBasedModel():
    basedModels = []
    basedModels.append(('LR'   , LogisticRegression()))
    basedModels.append(('LDA'  , LinearDiscriminantAnalysis()))
    basedModels.append(('KNN'  , KNeighborsClassifier()))
    basedModels.append(('CART' , DecisionTreeClassifier()))
    basedModels.append(('NB'   , GaussianNB()))
    basedModels.append(('SVM'  , SVC(probability=True)))
    basedModels.append(('AB'   , AdaBoostClassifier()))
    basedModels.append(('GBM'  , GradientBoostingClassifier()))
    basedModels.append(('RF'   , RandomForestClassifier()))
    basedModels.append(('ET'   , ExtraTreesClassifier()))

    
    return basedModels

def Eval_algo_models(X_train, y_train, models, scoring='accuracy', num_folds=10):
    """
        Description
        -----------
        Evaluate models with cross validation
        Parameters
        ----------
            X_train : Pandas data frame of features
            y_train : Pandas data frame of target
            models : list of models
            scoring : (default accuracy) scoring method
            num_folds : (default 10) number of folds for cross validation
        Returns
        -------
            names : list of model names
            results : list of model results
    """
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds)
        cv_results = cross_val_score(model, X_train, y_train.values.ravel(), cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    return names, results