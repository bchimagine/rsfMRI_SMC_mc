#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:35:41 2021

@author: ch208071
"""

import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import sklearn
import scipy
from numpy import linalg as LA
from MiscFuncForfMRIAnalysis import *
import csv


#path_to_save = '/fileserver/fetal/Arvind/fMRI/ABCD/mediummotion/sub-NDARINVC9CUGR6X/ses-baselineYear1Arm1/func/ForAnalysis/'
path_to_save = '/fileserver/fetal/Arvind/fMRI/ABCD/mediummotion/sub-NDARINVTPCLKWJ5/ses-baselineYear1Arm1/func/ForAnalysis/'
corr_matrix_path = ''

#ind_otherRuns = [0,1,2] # variable storing the indices corresponding to other runs; will change with the ABCD datasets
#ind_ref = 4 # index corresponding to the run, which is going to be ref. This will change with the ABCD dataset.
ind_otherRuns = [1,2,3] # this should be all runs except ind_ref, [1,2,3] is actrually going to be rest2,rest3 and rest4 (matrices in these folders).
ind_ref = 1 # this will be rest1 (matrix in rest1)
ref =  glob.glob(path_to_save + 'rest' + str(ind_ref) +  '/corrmatrix_numpy/' + '*_vvr_scr.npy')
corrMatrix_tr= np.load(ref[0])
sh = corrMatrix_tr.shape
VecFromUpperMat_tr =  GenerateVecFromMatForPlots(corrMatrix_tr,sh[0])
for i in ind_otherRuns:
    data_path = path_to_save + 'rest' + str(i+1) +  '/corrmatrix_numpy/' 
    npfname = data_path + '*.npy'
    for j in glob.glob(npfname):
       # print(j)
        NiftiFile = os.path.basename(j)
        #print(fmri_fname)
        f = os.path.splitext(NiftiFile)[0]
        g= os.path.splitext(f)[0]
        print(g)
        corrMatrix = np.load(j)
        sh = corrMatrix.shape
        VecFromUpperMat = GenerateVecFromMatForPlots(corrMatrix,sh[0])
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(VecFromUpperMat_tr ,  VecFromUpperMat)
        
       ## fitting a line between reference and motion/proposed correlations
       ## for different datasets the parameters of the RANSAC algorithm could change. Good to test different parameters
        ransac = sklearn.linear_model.RANSACRegressor(min_samples =5000 ,max_trials = 100)
        ransac.fit(VecFromUpperMat_tr.reshape(-1,1) , VecFromUpperMat.reshape(-1,1))
        ransac_slope = ransac.estimator_.coef_[0,0]
        ransac_intercept = ransac.estimator_.intercept_[0]
        pred = VecFromUpperMat_tr.reshape(-1,1)*ransac_slope  +  ransac_intercept 
        R2_sc = np.sqrt(sklearn.metrics.r2_score(VecFromUpperMat.reshape(-1,1), pred))
    
       
        print('The r value is ',R2_sc)
        print('The slope is',ransac_slope)
        print('The intercept is',ransac_intercept )
     
        plt.figure()
        plt.plot(VecFromUpperMat_tr,pred,'r',label = 'fitted line')
        plt.plot(VecFromUpperMat_tr,VecFromUpperMat_tr,linestyle = '--', dashes=(4,8),label =  'x=y') # plotting y=x line for reference
        plot2DHistogram(VecFromUpperMat_tr,VecFromUpperMat,25)
    
        plt.legend(prop={"size":12})
        plt.title("r value = "  + "{:.2f}".format(R2_sc) + ", slope = "  + "{:.2f}".format(ransac_slope) )
        fname =  path_to_save + 'rest' + str(i+1) + '/corrmatrix/'+ g + '_hmap' +  '.pdf'
        plt.savefig(fname)
        plt.show()
        
        ## saving the slope and intercept in a csv file.
        # myCsvRow = [str(R2_sc),str(ransac_slope),str(ransac_intercept)]
        # with open('document.csv','a') as params:
        #     #params.write(myCsvRow)
        #     writer = csv.writer(params)
        #     writer.writerow(myCsvRow )
            



        
            
        
        
        
        
        
      