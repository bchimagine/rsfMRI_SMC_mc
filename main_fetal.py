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
from MiscFuncForfMRIAnalysis import *



##1271s1
#Yref = '../fetal/1271s1/T2andfMRIreg/ori_vol1.nii.gz'
path_to_save = '../fetal/1311s1_65/analysis/'
Yref = '../fetal/1311s1_65/recon3D_noscrub/recon3D0000.nii.gz'
#Yref = '../fetal/1315s1_51/analysis/mean_reconfetal1315s1_51_masked.nii.gz'
fmri_fname = path_to_save + 'reconfetal1311s1_65_masked.nii.gz'

Ymov_atlas = path_to_save +  'atlas_t2final_1311s1.nii.gz'
mask = path_to_save + 'mask_1311s1_65.nii.gz'


#savepath = path_to_save+ 'original_masked.nii.gz'
#if np.allclose(nib.load(fmri_fname).affine,nib.load(mask).affine) !=  True:
#    ModifyAffine(fmri_fname,mask,savepath)
#    

Ypr = sitk.ReadImage(fmri_fname,sitk.sitkFloat64)
vx,vy,vz,TR = Ypr.GetSpacing()

###  Regressing out Nuisance signals.
reg_model =  'membrane_energy'
WarpedFile = path_to_save + 'T2inatlasspaceandfunc/FinalWarped.nii.gz'
FieldFile = path_to_save + 'T2inatlasspaceandfunc/WarpField.nii.gz'
FieldCoeffFile =  path_to_save + 'T2inatlasspaceandfunc/FieldCoeff.nii.gz'

outFile_WM = path_to_save + '/T2inatlasspaceandfunc/WMwarpedtofunc.nii.gz'
outFile_CSF = path_to_save + 'T2inatlasspaceandfunc/CSFwarpedtofunc.nii.gz'

ConfoudsCsv_path = path_to_save + "T2inatlasspaceandfunc/confounds.csv"

Y_WM = path_to_save + 'segmentedmasks/maskWM_atlas_t2.nii.gz' 
Y_CSF = path_to_save +  'segmentedmasks/maskCSF_atlas_t2.nii.gz'

seedpoints_csv = "../fetal/seedpointsinSTA/STAseeds_worldcoords_35.xlsx"

datatype= 'pr'
corr_cost = 'corr'

## Figure out inter-volume motion in the fmri data of interest
#if datatype == 'ori' or datatype == 'motion' or 'pr':
#    Y = sitk.ReadImage(fmri_fname,sitk.sitkFloat64)
#    fixed_mask = sitk.ReadImage(mask,sitk.sitkFloat64)
#    Ximg,params = RegisterVolsoffMRItoFirstVol(Y,fixed_mask,corr_cost)
#    params = np.insert(params,0,np.zeros([1,6]),0)
#    np.savetxt(path_to_save + 'motion_params_pr.txt',params) 
#else:
#    params = []
## Nusiance signals extraction
#params = np.loadtxt(path_to_save + 'motion_params.txt')
#params= np.insert(params,0,np.zeros([1,6]),0)
params=[]
ExtractingNuisanceSignals(Ymov_atlas,Yref,params,reg_model,fmri_fname,mask,corr_cost,WarpedFile,FieldFile,FieldCoeffFile,Y_WM,Y_CSF,outFile_WM,outFile_CSF, ConfoudsCsv_path,path_to_save, datatype)

## Transformt the atlas coordinates to the functional space./
Ymov_STA = '../fetal/STA35.nii.gz'
#Ymov_STA = '../fetal/1271s1/T2andfMRIreg/atlas_t2final_1271s1.nii.gz'
DOF = 6
params_6 = RegisterTwoVolumesUsingFLIRT(Ymov_STA,Yref,DOF,[],path_to_save)
print('FLIRt_6 done')
DOF = 12
params_12 = RegisterTwoVolumesUsingFLIRT(Ymov_STA,Yref,DOF,params_6,path_to_save)
ApplyFlirtTransform(Ymov_STA,Yref,params_12,path_to_save + 'STA_flirted.nii.gz')
#ApplyFlirtTransform(Ymov,Yref,params_12,'../fetal/1271s1/T2andfMRIreg/MNItomeannomo.nii.gz')
print('FLIRT_12 done')

#newseedpoint_csv = "../fetal/seedpointsinSTA/STAseeds_worldcoords_35_temp.xlsx"
SaveInPath = path_to_save + 'corrmatrix/'
df = pd.read_csv(seedpoints_csv,header=None)
coords = df.values

#coords = [(-16,-23,23 )]
#coords = [(-23.19,8.79,29.59)]
mapping = 'rigid'
coords_afterflirt = TransformCoordsFromSrcToTg(coords,Ymov_STA, Yref, params_12,mapping,[])
displayCoords(coords,coords_afterflirt,Ymov_STA,Yref,0)
print('done')

reg_model =  'membrane_energy'
WarpedFile = path_to_save + 'STAandfunc/FinalWarped.nii.gz'
FieldFile = path_to_save + 'STAandfunc/WarpField.nii.gz'
FieldCoeffFile = path_to_save +  'STAandfunc/FieldCoeff.nii.gz'
reg_model =  'membrane_energy'
RegisterTwoVolumesUsingFNIRT(Ymov_STA,Yref,reg_model,params_12,WarpedFile, FieldFile, FieldCoeffFile )
print('FNIRT done')

mapping = 'nonrigid'
coords_afterfnirt = TransformCoordsFromSrcToTg(coords,Ymov_STA, Yref, params_12,mapping,FieldFile)
displayCoords(coords,coords_afterfnirt,Ymov_STA,Yref,0)
print('done')

# Draw spheres around the coords
sm_sigma  = 6 # sigma for smoothing
r_sph = 10 # radius of the sphere
hp_cutoff = 0.01
lp_cutoff = 0.1
spheres_masker = CreateNiftiSpheresMasker(coords_afterfnirt,sm_sigma,r_sph,lp_cutoff,hp_cutoff,TR)

# Compute Correlation matrix
corr_matrix,NameOfFile = ComputeAndSaveCorrMatix(fmri_fname,spheres_masker,SaveInPath,ConfoudsCsv_path)

#corr_matrix,NameOfFile = ComputeAndSaveCorrMatix(fmri_fname,spheres_masker,SaveInPath,None)


### Preprocessing the fMRI data before running melodic 
#fmri_fname = '../fetal/1271s1/T2andfMRIreg/reconfetal1271s1.nii.gz'
#fmri_fname = '../fetal/1271s1/T2andfMRIreg/original.nii.gz'
#mask = '../fetal/1271s1/T2andfMRIreg/mask_1271s1_47.nii.gz'
#pathtosave = '../fetal/1271s1/T2andfMRIreg/ForMelodic/'
#hp_cutoff = 0.01
#lp_cutoff = 0.1
#
### Motion compensation for original 
#corr_cost = 'corr'
#Y = sitk.ReadImage(fmri_fname,sitk.sitkFloat64)
#fixed_mask = sitk.ReadImage(mask,sitk.sitkFloat64)
#Ximg,params = RegisterVolsoffMRItoFirstVol(Y,fixed_mask,corr_cost)
#sitk.WriteImage(Ximg,pathtosave + 'orig_mc_notprocessed.nii.gz')  
#
#fmri_fname = pathtosave + 'orig_mc_notprocessed.nii.gz'
#
#fmri_img  = PreProcessfMRIForMelodic(fmri_fname,mask,hp_cutoff,lp_cutoff,TR)
#sitk.WriteImage(fmri_img,pathtosave + 'orig_mc_' + 'processed.nii.gz')  

### Reguistering ori func to STA35
#Ymov = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regforigmcmaskedtoSTA35/mean_originalmc_masked.nii.gz'
#
#Yref = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regforigmcmaskedtoSTA35/STA35.nii.gz'
#path_to_save = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regforigmcmaskedtoSTA35/'
#DOF = 6
#params_6 = RegisterTwoVolumesUsingFLIRT(Ymov,Yref,DOF,[],path_to_save)
#print('FLIRt_6 done')
#DOF = 12
#params_12 = RegisterTwoVolumesUsingFLIRT(Ymov,Yref,DOF,params_6,path_to_save)
#print('FLIRT_12 done')
#ApplyFlirtTransform(Ymov,Yref,params_12,'../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regforigmcmaskedtoSTA35/STA_flirted.nii.gz')
#
## testing
#params = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regforigmcmaskedtoSTA35/params12funtoSTA_calc'
#ApplyFlirtTransform(Ymov,Yref,params,'../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regforigmcmaskedtoSTA35/func_flirted_new.nii.gz')
#
#
#Ymov =  '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/originalmc_masked.ica/filtered_func_data.ica/melodic_IC.nii.gz'
#path_to_save = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/originalmc_masked.ica/filtered_func_data.ica/'
#params = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regforigmcmaskedtoSTA35/params12funtoSTA_calc'
#ApplyFlirtTransform(Ymov,Yref,params,'../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/originalmc_masked.ica/filtered_func_data.ica/melodic_flirted.nii.gz')
#
#
### Reguistering pr_noscrub func to STA35
#Ymov = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regprnoscrubtoSTA35/mean_prnoscrub_masked.nii.gz'
#
#Yref = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regprnoscrubtoSTA35/STA35.nii.gz'
#path_to_save = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regprnoscrubtoSTA35/'
#DOF = 6
#params_6 = RegisterTwoVolumesUsingFLIRT(Ymov,Yref,DOF,[],path_to_save)
#print('FLIRt_6 done')
#DOF = 12
#params_12 = RegisterTwoVolumesUsingFLIRT(Ymov,Yref,DOF,params_6,path_to_save)
#print('FLIRT_12 done')
#ApplyFlirtTransform(Ymov,Yref,params_12,'../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regprnoscrubtoSTA35/STA_flirted.nii.gz')
#
### testing
#Ymov =  '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/prnoscrub_masked.ica/filtered_func_data.ica/melodic_IC.nii.gz'
#path_to_save = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/prnoscrub_masked.ica/filtered_func_data.ica/'
#params = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regprnoscrubtoSTA35/params12functoSTA_calc'
#ApplyFlirtTransform(Ymov,Yref,params,'../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/prnoscrub_masked.ica/filtered_func_data.ica/melodic_flirted.nii.gz')

#
#
#Ymov =  '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/originalmc_masked.ica/filtered_func_data.ica/melodic_IC.nii.gz'
#path_to_save = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/originalmc_masked.ica/filtered_func_data.ica/'
#params = '../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/regforigmcmaskedtoSTA35/params12funtoSTA_calc'
#ApplyFlirtTransform(Ymov,Yref,params,'../fetal/1271s1/T2andfMRIreg/ForMelodic/Melodicprocessing/originalmc_masked.ica/filtered_func_data.ica/melodic_flirted.nii.gz')


#c=0
#np_coords = np.array(coords)
#display_src=plotting.plot_anat(Ymov,display_mode = 'ortho',cut_coords = (np_coords[c,0],np_coords[c,1],np_coords[c,2]))
#display_src.add_markers(marker_coords=[[np_coords[c,0],np_coords[c,1],np_coords[c,2]]], marker_color='g',marker_size=75)
#


## using ITK
#ref =  sitk.ReadImage(Yref,sitk.sitkFloat64)
#moving =  sitk.ReadImage(Ymov,sitk.sitkFloat64)
#m =  sitk.ReadImage(mask,sitk.sitkFloat64)
#tf = registerTwoVolumesUsingITK(moving,ref,[],'corr')
#tfm = []
#tfm.append(tf.GetParameters())
#Ximg = sitk.Resample(moving,ref,tf,sitk.sitkLanczosWindowedSinc,0.0,moving.GetPixelID())
#sitk.WriteImage(Ximg,'../fetal/1300s2/T2andfMRIreg/T2regtofMRI_ITK.nii.gz')  
#np.savetxt('../fetal/1300s2/T2andfMRIreg/paramsT2tofMRI_ITK.txt',np.asarray(tfm)) 

#MNI
#Yref = '../fetal/1271s1/T2andfMRIreg/test/testingflirt/mean_nomotion_LAS.nii.gz'
##Yref = sitk.ReadImage('../fetal/1271/T2andfMRIreg/reconfetal_vol1_masked.nii.gz',sitk.sitkFloat64)
##Ymov = '../fetal/1271s1/T2andfMRIreg/t2_t2_1271s1_rad.nii.gz'
##Ymov = '../fetal/1271s1/T2andfMRIreg/atlas_t2final_1271s1.nii.gz'
#Ymov = '../fetal/1271s1/T2andfMRIreg/test/testingflirt/MNI152_T1_2mm.nii.gz'
#path_to_save = '../fetal/1271s1/T2andfMRIreg/test/testingflirt/'
