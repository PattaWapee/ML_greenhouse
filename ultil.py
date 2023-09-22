import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import xlrd
from datetime import datetime


##############################
####### missing ratio ########
##############################

def cal_missing_ratio(df):
    """
        Description
        -----------
        Calculate missing ratio for each columns of df
        Parameters
        ----------
            df : Pandas data frame 
        Returns
        -------
            missing_df : Pandas data frame of missing value ratio with index of column name
    """
    missing_ratio = df.isnull().sum()/len(df)
    missing_ratio = missing_ratio[missing_ratio > 0]
    missing_df = pd.DataFrame(missing_ratio[missing_ratio > 0], columns=[
                              'missing_value_ratio'])
    return missing_df


def plot_missing_ratio(df, figsize=(10, 5)):
    """
        Description
        -----------
        Plot missing ratio for each columns of df
        Parameters
        ----------
            df : Pandas data frame of missing ratio
        Returns
        -------
            Showing plot
    """
    missing_df = cal_missing_ratio(df)
    fig, ax = plt.subplots(figsize=figsize)
    missing_df.plot(kind='bar', ax=ax)
    plt.title('Missing value ratio')
    return ax

################################
####### data details ###########
################################

def plot_nunique(df, col_list):
    """
        Description
        -----------
        Plot number of unique entries in column list
        Parameters
        ----------
            df : Pandas data frame
            col_list : list of column to plot unique entries
        Returns
        -------
            Showing plot
    """
    df[col_list].nunique().plot(kind='bar')
    plt.title('Number of unique entries')
    plt.show()


def plot_topN(df, col, N):
    """
        Description
        -----------
        Plot Top 10 of unique entries in column
        Parameters
        ----------
            df : Pandas data frame
            col : column to plot unique entries
            N : top N to plot
        Returns
        -------
            Showing plot
    """
    df_count = df[col].value_counts().sort_values(ascending=False)
    df_count[0:N+1].plot(kind='bar')
    plt.xlabel('entry name')
    plt.ylabel('number of entries')
    plt.title('Top ' + str(N) + ' entries')


def plot_corr(df, size=20, vmin=-1, vmax=1, center=0):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    fig, ax = plt.subplots(figsize=(size, size))
    ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=vmax, vmin=vmin, center=center,
                     square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return ax


def match_columns(df, col1, col2, normalize=True):
    """
        Description
        -----------
        Compare two columns in dataframe and return count of match and no match)
        Parameters
        ----------
            df : Pandas data frame
            col1 : column 1 to compare
            col2 : column 2 to compare
        Returns
        -------
            sdf : Pandas data frame of count of match and no match
    """
    sdf = (df[col1].eq(df[col2])
           .value_counts(normalize=normalize)
           .rename({True: 'mach', False: 'no match'})
           .rename_axis('state')
           .reset_index(name='count ratio'))
    return sdf


##############################
####### save and load ########
##############################


def read_pickle_obj(pickle_file):
    open_file = open(pickle_file, "rb")
    data = pickle.load(open_file)
    open_file.close()
    return(data)


def save_pickle_obj(file_name, obj_file):
    open_file = open(file_name, "wb")
    pickle.dump(obj_file, open_file)
    open_file.close()

##############################
###### Converting ############
##############################

def col_to_datetime(df, col, type=None, drop= False,index_time=False):
    """
        Description
        -----------
        Convert date column to datetime
        Parameters
        ----------
            df : Pandas data frame
            col : column to convert to datetime
        Returns
        -------
            df : Pandas data frame with converted column
    """
    #if type == None:
    #    df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
    if type == 'xlrd':
        df[col] = df[col].apply(lambda x: datetime(*xlrd.xldate_as_tuple(x, 0)))

    else:
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)

    if index_time == True:
        df.index = df[col]

    if drop == True:
        df.drop(col, axis=1, inplace=True)    
    
    return df