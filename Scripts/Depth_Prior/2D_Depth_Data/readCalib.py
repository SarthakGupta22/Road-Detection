# Read the calibration file and print the parameters
import os
import sys
import numpy as np

def getMatrices(file_name):
    calibFile = open(file_name, "rt")
    
    content = calibFile.read()
    
    split_content = content.split('\n')
    
    K_params = str(split_content[0])
    T_params = str(split_content[5])
    R_params = str(split_content[4])
    K_params = K_params.split(" ")
    T_params = T_params.split(" ")
    R_params = R_params.split(" ")    
    #print(K_params)
    
    T_list = []
    R_list = []
    K_list = []

    for i in range(1, len(K_params)):
        K_list.append(float(K_params[i]))

    for i in range(1,len(T_params)):
        T_list.append(float(T_params[i]))

    for i in range(1, len(R_params)):
        R_list.append(float(R_params[i]))

    T = np.asanyarray(T_list).reshape(3,4)
    K = np.asanyarray(K_list).reshape(3,4)
    R = np.asanyarray(R_list).reshape(3,3)


    return T,K[:,0:3],R
