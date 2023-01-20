#import packages from different Modules 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
from sklearn.neighbors import  NearestNeighbors #Importing a library for finding a nearest neighbors for DBSCAN.
from sklearn.cluster import DBSCAN  # module from sklearn. cluster to perfome a density based clustering using DBSCAN alogorithm
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.metrics import silhouette_score 
from sklearn import preprocessing
import seaborn as sns

def exp_growth(t, scale, growth):
    f = scale * np.exp(growth * (t-1960))
    return f
def logistics(t, scale, growth, t0):
    ''' 
        Computes logistics function with scale, growth raat
        and time of the turning point as free parameters
    '''
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f
def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   
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

def applyDBSCANalgo(dataframe, min_sample_pre, col_Name_1, col_Name_2):
    '''
    DBSCAN is an algoritm for Density-based clustering.Using density based clustering we find a noise or outliner from a given data
    Using applyDBSCANalgo() pass dataframe as parameter and then claculated a nearestneighbour to calculate 
    eps and min_sample which are parameter for DBSCAN and ploting  result to scatter plot, calculate number of clusters.
    col_Name: Name of column to use for ploting
    '''
    # Extract columns from dataframe to process a clustering
    dataframe_X = dataframe.loc[:,[col_Name_1, col_Name_2]].values
    print('Shape of DF: \n', dataframe_X.shape)

    dataframe_X = StandardScaler().fit_transform(dataframe_X)
    # Creating an object of the NearestNeighbors class and fitting the data to the object
    #n_neighbors : To seek nearestneighobor
    nearestneighbor = NearestNeighbors(n_neighbors=min_sample_pre + 1).fit(dataframe_X)
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
    fig, ax = plt.subplots()
    sc = ax.scatter(dataframe[col_Name_2], dataframe[col_Name_1], c = labels, cmap= "plasma")
    ax.legend(*sc.legend_elements(), title = 'cluster')
    plt.xlabel(col_Name_2)
    plt.ylabel(col_Name_1)
    plt.title('Density-based(DBSCAN algorithm) Clustering.')

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

grp_average_by_year = rmvna_access_electricity_df.groupby('Years', as_index=False).mean()

plt.figure()
plt.plot(grp_average_by_year['Years'], grp_average_by_year['Access to electricity (% of population)'], linestyle='dashed')
plt.title('Access to elecricity average.')
plt.xlabel('Years')
plt.ylabel('Access to electricity (% of population)')
plt.xticks(grp_average_by_year['Years'], rotation=90)
plt.show()

# Group a data by year and calculated yearly average of data.
grp_access_electricity_per_popu = rmvna_access_electricity_df.groupby('Years', as_index=False).mean()
print('Group data set: \n', grp_access_electricity_per_popu.head(10))

# call applyDBSCANalgo() to perfome density-based clustering
applyDBSCANalgo(grp_access_electricity_per_popu,10,'Years','Access to electricity (% of population)')

# Changed Data type for column years object ot int
rmvna_access_electricity_df['Years'] = rmvna_access_electricity_df['Years'].astype(int)
print('\n Data type of dataFrame: ',rmvna_access_electricity_df.dtypes)

#fit exponential  growth
popt, covar = opt.curve_fit(exp_growth, rmvna_access_electricity_df['Years'],rmvna_access_electricity_df['Access to electricity (% of population)'])
print('Fit parameter: ', popt) #Fit parameter:  [-1.05093743e-20  9.99999993e-01]

# use *popt to pass on the fit parameters
rmvna_access_electricity_df['pop_exp'] = exp_growth(rmvna_access_electricity_df['Years'], *popt)
plt.figure()
plt.plot(rmvna_access_electricity_df['Years'], rmvna_access_electricity_df['Access to electricity (% of population)'], label = 'data')
plt.plot(rmvna_access_electricity_df['Years'], rmvna_access_electricity_df['pop_exp'], label='fit')
plt.title("First fit")
plt.xlabel('year')
plt.ylabel('Access to electricity (% of population)')
print()

# find a feasible start value the pedestrian way
# the scale factor is way too small. The exponential factor too large.
# Try scaling with the 1950 population and a smaller exponential factor
# decrease or increase exponential factor until rough agreement is reached
# growth of 0.02 gives a reasonable start value
popt = [4e1, 0.05]
rmvna_access_electricity_df["pop_exp"] = exp_growth(rmvna_access_electricity_df["Years"], *popt)
plt.figure()
plt.plot(rmvna_access_electricity_df['Years'], rmvna_access_electricity_df['Access to electricity (% of population)'], label = 'data')
plt.plot(rmvna_access_electricity_df['Years'], rmvna_access_electricity_df['pop_exp'], label='fit')
plt.legend()
plt.xlabel('year')
plt.ylabel('Access to electricity (% of population)')
plt.title("Improved start value")
print()

# fit exponential growth
popt, covar = opt.curve_fit(exp_growth, rmvna_access_electricity_df["Years"],
rmvna_access_electricity_df["Access to electricity (% of population)"], p0=[4e1, 0.6])

# much better
print("Fit parameter", popt)
rmvna_access_electricity_df["pop_exp"] = exp_growth(rmvna_access_electricity_df["Years"], *popt)

plt.figure()
plt.plot(rmvna_access_electricity_df['Years'], rmvna_access_electricity_df['Access to electricity (% of population)'], label = 'data')
plt.plot(rmvna_access_electricity_df['Years'], rmvna_access_electricity_df['pop_exp'], label='fit')
plt.legend()
plt.xlabel('year')
plt.ylabel('Access to electricity (% of population)')
plt.title("Final fit exponential growth")
print()

#Logistics for of curve_fit
popt, covar = opt.curve_fit(logistics, rmvna_access_electricity_df['Years'], rmvna_access_electricity_df["Access to electricity (% of population)"],
p0=(2e9, 0.05, 1990.0))
print("Fit parameter", popt)
rmvna_access_electricity_df["pop_log"] = logistics(rmvna_access_electricity_df["Years"], *popt)
plt.figure()
plt.title("logistics function")
plt.plot(rmvna_access_electricity_df['Years'], rmvna_access_electricity_df['Access to electricity (% of population)'], label = 'data')
plt.plot(rmvna_access_electricity_df['Years'], rmvna_access_electricity_df['pop_exp'], label='fit')
plt.legend()
plt.xlabel('year')
plt.ylabel('Access to electricity (% of population)')

# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)
low, up =err_ranges(rmvna_access_electricity_df["Years"], logistics, popt, sigma)
plt.figure()
plt.title("logistics function")
plt.plot(rmvna_access_electricity_df['Years'], rmvna_access_electricity_df['Access to electricity (% of population)'], label = 'data')
plt.plot(rmvna_access_electricity_df['Years'], rmvna_access_electricity_df['pop_exp'], label='fit')
plt.fill_between(rmvna_access_electricity_df["Years"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("population")

print("Forcasted population")
low, up = err_ranges(2030, logistics, popt, sigma)
print("2030 between ", low, "and", up)
low, up = err_ranges(2040, logistics, popt, sigma)
print("2040 between ", low, "and", up)
low, up = err_ranges(2050, logistics, popt, sigma)
print("2050 between ", low, "and", up)

print("Forcasted population")
low, up = err_ranges(2030, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err_ranges(2040, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err_ranges(2050, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)
plt.show()