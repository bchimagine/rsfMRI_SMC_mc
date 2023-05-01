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
from MiscFuncForfMRIAnalysis import *

##1271s1
#Yref = '../fetal/1271s1/T2andfMRIreg/ori_vol1.nii.gz'
params_path = '/fileserver/fetal/Arvind/fMRI/ABCD/highmotion/sub-NDARINVB1E3XW7P/ses-baselineYear1Arm1/func/motionparams/'
path_to_save = '/fileserver/fetal/Arvind/fMRI/ABCD/highmotion/sub-NDARINVB1E3XW7P/ses-baselineYear1Arm1/func/ForAnalysis/'
Ystruct = '/fileserver/fetal/Arvind/fMRI/ABCD/highmotion/sub-NDARINVB1E3XW7P/ses-baselineYear1Arm1/anat/sub-NDARINVB1E3XW7P_ses-baselineYear1Arm1_rec-normalized_run-02_T1w_LAS.nii.gz' # ref1 for now
MNI_template = '/fileserver/fetal/Arvind/fMRI/ABCD/MNI152_T1_2mm.nii.gz' # moving

Y_WM =  '/fileserver/fetal/Arvind/fMRI/ABCD/SegmentationMNI/MNI2mm/MNI_BET_seg_2.nii.gz' 
Y_CSF = '/fileserver/fetal/Arvind/fMRI/ABCD/SegmentationMNI/MNI2mm/MNI_BET_seg_0.nii.gz' 
coords = CoordsFromPowerAtlas() # Coords from Power Atlas

corr_cost = 'corr'
substr = '_vvr'
sm_sigma  = 6 # sigma for smoothing
r_sph = 10 # radius of the sphere
hp_cutoff = 0.01
lp_cutoff = 0.1
#data_path = '/home/ch208071/Research/Codes/fMRI/rsfMRIwithandwithoutmotion/Matlabcodes/SLR_IRLS/MainCodes_modelincludesmotion/newformulation/experiments/vol_based/ABCD/sub-NDARINVC9CUGR6X/rest1/'
for i in range(4):
    data_path = path_to_save + 'rest' + str(i+1) + '/' # change str(1) to str(i) in a loop/
    files = glob.glob(data_path + 'sub*.nii.gz')
    for j in files:
        fMRI_fname = j
        NiftiFile = os.path.basename(fMRI_fname)
        #print(fmri_fname)
        f = os.path.splitext(NiftiFile)[0]
        g= os.path.splitext(f)[0]
        mask = data_path + 'mean_' + g + '_masked_mask.nii.gz'
        #print(mask)
        Ypr = sitk.ReadImage(fMRI_fname,sitk.sitkFloat64)
        vx,vy,vz,TR = Ypr.GetSpacing()
        ExtractFirstVolFrom4DfMRItimeseries(fMRI_fname,data_path) # First vol used as reference in registration.
        Yref = data_path  + 'vol1.nii.gz'
        
            ###  Regressing out Nuisance signals.
        reg_model =  'membrane_energy'
        WarpedFile = data_path + 'srcintg/'  + 'FinalWarped_' + g + '.nii.gz'
        FieldFile = data_path + 'srcintg/'  + 'WarpField_' + g + '.nii.gz'
        FieldCoeffFile =  data_path + 'srcintg/'  + 'FieldCoeff_' + g + '.nii.gz'
    
        outFile_WM = data_path + 'srcintg/'  + 'WMwarpedtofunc_' + g + '.nii.gz'
        outFile_CSF = data_path + 'srcintg/'  + 'CSFwarpedtofunc_' + g + '.nii.gz'
    
        ConfoudsCsv_path = data_path + 'srcintg/' + 'confounds_' + g + '.csv'
    
    ## Nusiance signals extraction
        params_string = str(i+1)
        params_string = params_string.zfill(2)
        paramsFile  = glob.glob(params_path + '*_run-' + params_string + '*.txt')
        
        
        params = np.loadtxt(paramsFile[0])
        params= np.insert(params,0,np.zeros([1,6]),0)
        
        ExtractingNuisanceSignals(MNI_template,Yref,params,reg_model,fMRI_fname,mask,corr_cost,WarpedFile,FieldFile,FieldCoeffFile,Y_WM,Y_CSF,outFile_WM,outFile_CSF,ConfoudsCsv_path,data_path)
        print('csv file created')
        
        ## Transformt the atlas coordinates to the functional space./
        if os.path.isfile(path_to_save  + 'MNItoStruct/SourceToStruct_params12')==True:
            params_srctostruct = path_to_save  + 'MNItoStruct/SourceToStruct_params12'
        else:
            reg_option = 'nonrigid'
            RegisterSourceToStructural(MNI_template, Ystruct,path_to_save,reg_option)
            params_srctostruct = path_to_save  + 'MNItoStruct/SourceToStruct_params12'
            ApplyFlirtTransform(MNI_template,Ystruct,params_srctostruct ,path_to_save  + 'MNItoStruct/MNI_flirtedToStruct.nii.gz')
            
        reg_option = 'nonrigid'
        RegisterStructuralToTarget(Ystruct,Yref,data_path,reg_option,g)
        params_structtotg = data_path + g +  '_StructToTarget_params12'
        ApplyFlirtTransform(Ystruct,Yref,params_structtotg,data_path + 'StructflirtedToTarget_' + g  + '.nii.gz')
        
        #coords = [(39.5,19.3946,-5.76 )]
        mapping = 'nonrigid'
        FieldFile_SourceToStruct = path_to_save +  'MNItoStruct/WarpField_SourceToStruct.nii.gz' # when option is rigid, pass []
        FieldFile_StructToTarget = data_path + 'WarpField_StructToTarget_' +  g +  '.nii.gz'   # when option is rigid, pass []
        coords_afterfnirt = TransformCoordsFromSrcToTgViaStructural(coords,MNI_template,Yref,Ystruct,params_srctostruct,params_structtotg,FieldFile_SourceToStruct,FieldFile_StructToTarget,mapping)
        displayCoords(coords,coords_afterfnirt,MNI_template,Yref,10)
        print('coordinates transformed to functional space')
        
        # Draw spheres around the coords

        spheres_masker = CreateNiftiSpheresMasker(coords_afterfnirt,sm_sigma,r_sph,lp_cutoff,hp_cutoff,TR)

    # Compute Correlation matrix
        SaveInPath =  data_path + 'corrmatrix/'
        corr_matrix,NameOfFile = ComputeAndSaveCorrMatix(fMRI_fname,spheres_masker,SaveInPath,ConfoudsCsv_path)
        corr_numpy_fname = os.path.join(data_path + 'corrmatrix_numpy' ,NameOfFile)
        np.save(corr_numpy_fname,corr_matrix)
        
        if g.find(substr) != -1:
            if  os.path.isfile(data_path + 'vol_rem_thr0.5_afterscrubbing.npy')==True:
                vols = np.load(data_path + 'vol_rem_thr0.5_afterscrubbing.npy')
                Nvol= [int(x) for x in vols]
                Nvol  = np.array(Nvol)-1
                timeseries =  spheres_masker.fit_transform(fMRI_fname,confounds= ConfoudsCsv_path)
                timeseries_volscr = timeseries[Nvol,:] 
                corrmatrix_fname = os.path.join(SaveInPath,g + '_thr0.5_volscrubbed' +  '.pdf')
                corrmatrix_volscr = computecorrdirectly(timeseries_volscr,timeseries_volscr,corrmatrix_fname)
                corr_numpy_fname = os.path.join(data_path + 'corrmatrix_numpy' ,NameOfFile + '_scr')
                np.save(corr_numpy_fname,corr_matrix)
#                
        
    # Analyzing reconstructed images.
    files = glob.glob(data_path + 'recon*.nii.gz')
    #files = glob.glob(data_path + 'recon_*sub-NDARINVC9CUGR6X_ses-baselineYear1Arm1_task-rest_run-01_bold_*.nii.gz')
    for j in files:
        fMRI_fname = j
        NiftiFile = os.path.basename(fMRI_fname)
        #print(fmri_fname)
        f = os.path.splitext(NiftiFile)[0]
        g= os.path.splitext(f)[0]
        mask = data_path + 'mean_' + g + '_masked_mask.nii.gz'
        #print(mask)
        Ypr = sitk.ReadImage(fMRI_fname,sitk.sitkFloat64)
        vx,vy,vz,TR = Ypr.GetSpacing()
        ExtractFirstVolFrom4DfMRItimeseries(fMRI_fname,data_path) # First vol used as reference in registration.
        Yref = data_path  + 'vol1.nii.gz'
        
            ###  Regressing out Nuisance signals.
        reg_model =  'membrane_energy'
        WarpedFile = data_path + 'srcintg/'  + 'FinalWarped_' + g + '.nii.gz'
        FieldFile = data_path + 'srcintg/'  + 'WarpField_' + g + '.nii.gz'
        FieldCoeffFile =  data_path + 'srcintg/'  + 'FieldCoeff_' + g + '.nii.gz'
    
        outFile_WM = data_path + 'srcintg/'  + 'WMwarpedtofunc_' + g + '.nii.gz'
        outFile_CSF = data_path + 'srcintg/'  + 'CSFwarpedtofunc_' + g + '.nii.gz'
    
        ConfoudsCsv_path = data_path + 'srcintg/' + 'confounds_' + g + '.csv'
    
    ## Nusiance signals extraction
        params = [] ## For now motion params not regressed from recons.
        
        ExtractingNuisanceSignals(MNI_template,Yref,params,reg_model,fMRI_fname,mask,corr_cost,WarpedFile,FieldFile,FieldCoeffFile,Y_WM,Y_CSF,outFile_WM,outFile_CSF,ConfoudsCsv_path,data_path)
        print('csv file created')
        
       ## Transformt the atlas coordinates to the functional space./
        if os.path.isfile(path_to_save  + 'MNItoStruct/SourceToStruct_params12')==True:
            params_srctostruct = path_to_save  + 'MNItoStruct/SourceToStruct_params12'
        else:
            reg_option = 'nonrigid'
            RegisterSourceToStructural(MNI_template, Ystruct,path_to_save,reg_option)
            params_srctostruct = path_to_save  + 'MNItoStruct/SourceToStruct_params12'
            ApplyFlirtTransform(MNI_template,Ystruct,params_srctostruct ,path_to_save  + 'MNItoStruct/MNI_flirtedToStruct.nii.gz')
            
            
        reg_option = 'nonrigid'
        RegisterStructuralToTarget(Ystruct,Yref,data_path,reg_option,g)
        params_structtotg = data_path + g +  '_StructToTarget_params12'
        ApplyFlirtTransform(Ystruct,Yref,params_structtotg,data_path + 'StructflirtedToTarget_' + g  + '.nii.gz')
        
        #coords = [(39.5,19.3946,-5.76 )]
        mapping = 'nonrigid'
        FieldFile_SourceToStruct = path_to_save +  'MNItoStruct/WarpField_SourceToStruct.nii.gz' # when option is rigid, pass []
        FieldFile_StructToTarget = data_path + 'WarpField_StructToTarget_' +  g +  '.nii.gz'   # when option is rigid, pass []
        coords_afterfnirt = TransformCoordsFromSrcToTgViaStructural(coords,MNI_template,Yref,Ystruct,params_srctostruct,params_structtotg,FieldFile_SourceToStruct,FieldFile_StructToTarget,mapping)
        displayCoords(coords,coords_afterfnirt,MNI_template,Yref,10)
        print('coordinates transformed to functional space')
            
        # Draw spheres around the coords

        spheres_masker = CreateNiftiSpheresMasker(coords_afterfnirt,sm_sigma,r_sph,lp_cutoff,hp_cutoff,TR)

    # Compute Correlation matrix
        SaveInPath = data_path + 'corrmatrix/'
        corr_matrix,NameOfFile = ComputeAndSaveCorrMatix(fMRI_fname,spheres_masker,SaveInPath,ConfoudsCsv_path)
        
        corr_numpy_fname = os.path.join(data_path + 'corrmatrix_numpy' ,NameOfFile)
        np.save(corr_numpy_fname,corr_matrix)

        
    




