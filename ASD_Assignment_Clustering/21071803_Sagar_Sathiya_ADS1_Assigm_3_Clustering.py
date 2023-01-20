#import packages from different Modules 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import  NearestNeighbors #Importing a library for finding a nearest neighbors for DBSCAN.
from sklearn.cluster import DBSCAN  # module from sklearn. cluster to perfome a density based clustering using DBSCAN alogorithm
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.metrics import silhouette_score 
from sklearn import preprocessing
import seaborn as sns


def getDirecotry_path():
    '''
        This function is fetch a directory file path using os module and return a csvfile path.
    '''
    #get Current directory path
    current_direcotry_path = os.getcwd()
    print('Current direcotry path: ', current_direcotry_path) # Current direcotry path:s:\Visual_Studio\ASD_Assignment_Clustering

    #Change directory from current to previous
    os.chdir('..')
    chg_file_path = os.getcwd()
    print('csvFiles  direcotry path: ', chg_file_path) # csvFiles  direcotry path:  s:\Visual_Studio

    #change direcotory from current location to 'csvFiles'
    os.chdir('csvFiles')
    csv_file_path = os.getcwd()
    print('CSV filepath: ', csv_file_path) # CSV filepath:  s:\Visual_Studio\csvFiles
    
    return csv_file_path
    
print('CSV_File_path: \n', getDirecotry_path())
def readFilefromlocation(filename) :
    print('CSV_File_path: \n', getDirecotry_path()+'\\'+filename)
    '''
        This function is use for read file from "csvFiles" direcotry.After a reading file drop column and processed dataset 
        into well structured formate with header and returning a two data frame: orignal_data and t_dataframe
        orignal_data: This dataframe is return a data from a daset file.
        t_dataframe: This dataframe retruns a specific columns like year, and Country with a values.
        value_Name_indicator: Name of indicator from a dataset.
        indicator_Code: indicator code from a dataset.
    '''
    dataframe = pd.read_csv(getDirecotry_path()+'\\'+filename, on_bad_lines='skip')
    orignal_data = dataframe
    value_Name_indicator = dataframe['Indicator Name'][0]
    indicator_Code  = dataframe['Indicator Code'][0]
    dataframe = dataframe.drop(['Country Code', 'Indicator Name','Indicator Code'], axis = 1)
    years = list(dataframe.columns)

    # Reshaping a data unpivots dataframe from wide format to long format
    t_dataframe = pd.melt(dataframe, id_vars=['Country Name'], value_vars=years, var_name='Years', value_name=value_Name_indicator)
    return t_dataframe, orignal_data 

def dataCleaning(dataframe):
    '''
        This function is use for get information of dataframe.Moreover we calculat count of NaN value,Once NaN value found 
        drop NaN value or null value from a dataframe. Assign a cleaned data to cleanDataframe variable and 
        return a cleaned data frame.
        dataframe: Dataframe as parameter.
        cleandDataframe: Variable which has NaN value removed and return to function.
    '''
    # Information about dataframe
    print("Information of Transpose Data: ", dataframe.info())

    # Calculate Null or NaN value into dataset
    print('Total number of NaN count in dataset: ', dataframe.isna().sum())

    # Drop all NaN value and assign new dataframe.
    cleanDataframe = dataframe.dropna()
    
    return cleanDataframe
def applyDBSCANalgo(dataframe, min_sample_pre,col_Name):
    '''
    DBSCAN is an algoritm for Density-based clustering.Using density based clustering we find a noise or outliner from a given data
    Using applyDBSCANalgo() pass dataframe as parameter and then claculated a nearestneighbour to calculate 
    eps and min_sample which are parameter for DBSCAN and ploting  result to scatter plot, calculate number of clusters.
    col_Name: Name of column to use for ploting
    '''
    # Extract columns from dataframe to process a clustering
    #dataframe_X = dataframe.drop(col_Name, axis=1).values
    dataframe_X = np.array(dataframe[col_Name]).reshape(1,-1)
    dataframe_X = StandardScaler().fit_transform(dataframe_X)
    # Creating an object of the NearestNeighbors class and fitting the data to the object
    #n_neighbors : To seek nearestneighobor
    nearestneighbor = NearestNeighbors(n_neighbors=5).fit(dataframe_X)
    # Finding a nearest neighbors 
    distances, indices = nearestneighbor.kneighbors(dataframe_X)
    print('Distanced: and indices: ', distances, indices)

    # Sort Distances with min_sample_pre nearestneighbor and plot the variation
    distances = np.sort(distances[:min_sample_pre], axis=0)
    distances = distances[:,1]

    # KneeLocator to Detect Elbow point
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    eps_cal = float("{:.2f}".format(distances[knee.knee]))
    print('eps_cal: ', eps_cal)

    # Predicated value of min_sample from nearestneighbors
    dbscan = DBSCAN(eps = eps_cal, min_samples = min_sample_pre).fit(dataframe_X)
    dataframe_X_pred = dbscan.fit_predict(dataframe_X)
    labels = dbscan.labels_


    # Number of clusters in a labels, and noise
    n_clusters = len(set(labels))
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters :', n_clusters)
    print('Estimated number of noise points:' , n_noise_)
    print(f"silhouette_score: {silhouette_score(dataframe_X,labels)}")

    # Scatter plots
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.scatter(dataframe_X[:,0], dataframe_X[:,1], c=dataframe[col_Name], cmap="jet", edgecolor="None", alpha=0.35)
    ax1.set_title("Actual clusters")
    ax2.scatter(dataframe_X[:,0], dataframe_X[:,1], c=dataframe_X_pred, cmap="jet", edgecolor="None", alpha=0.35)
    ax2.set_title("DBSCAN clustering plot")
    plt.show()

# https://data.worldbank.org/indicator/EG.ELC.ACCS.ZS
# Access to electricity (% of population)
t_access_electricity_df, orignal_access_electricity_df = readFilefromlocation('API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_4771419.csv')
print('\n Transpose Access to Electricity (% of population): \n', t_access_electricity_df.head())
print('\n Orignal Access to Electricity (% of population): \n', orignal_access_electricity_df.head())

# Calling dataCleaning function and pass t_access_electricity_df as parameter
rmvna_access_electricity_df = dataCleaning(t_access_electricity_df)

# Recalculate a NaN presenet after droping from a dataframe
print('Re-checking NaN count: \n' , rmvna_access_electricity_df.isna().sum())

#Perform normalization on Dataframe to get ṇnormalized data
per_population_acc_elect_arr = np.array(rmvna_access_electricity_df['Access to electricity (% of population)']) 
norm_access_electricity_arr  = preprocessing.normalize([per_population_acc_elect_arr])
print('Normalized Value: ', norm_access_electricity_arr)

#list_df = {'Scaled_Access_to_electricity_per_population' : list(norm_access_electricity_arr)}
scaled_access_electricity_df = pd.DataFrame(data=per_population_acc_elect_arr, columns=['Scaled_Access_to_electricity_per_population'])
print('\n Scaled Data: \n', scaled_access_electricity_df.head(10))

applyDBSCANalgo(rmvna_access_electricity_df,10,'Access to electricity (% of population)')
