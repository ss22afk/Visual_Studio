#Package import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os

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
    
def readFilefromlocation(filename) :
    '''
    This function is use for read file from "csvFiles" directory. 
    After a reading file processa load file and retrun a dataframe
    filename: Name of file as parameter.
    '''
    
    dataframe = pd.read_csv(filename)

    return dataframe
