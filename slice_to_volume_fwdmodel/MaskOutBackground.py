#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:56:14 2020

@author: ch208071
"""
#from FuncsForfMRIanalysis import *
from nilearn import image,datasets,plotting,input_data,signal
import nibabel as nib
from nilearn.image import index_img,load_img
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask
from nilearn import masking
import numpy as np
import SimpleITK as sitk
import os
import glob



def numpy4Dtositk(X_np,origin,direction,spacing,nv):
    
    Xtemp = []
    
    for k in range(0,nv):
        Xtemp.append(sitk.GetImageFromArray(X_np[k,:,:,:]))
        
    Y = sitk.JoinSeries(Xtemp)
    Y.SetOrigin(origin)
    Y.SetSpacing(spacing)
    Y.SetDirection(direction) 
    
    return Y


def maskoutbg(fmri_fname,outFile):

    Y = sitk.ReadImage(fmri_fname,sitk.sitkFloat64);
    origin_4d = Y.GetOrigin()
    spacing_4d = Y.GetSpacing()
    direction_4d = Y.GetDirection()
    #nv = 120 # change nv depending on the dataset
    #nv = 160;
    n1,n2,nsl,nv = Y.GetSize()
    
    #fmri_fname1 = '/home/ch208071/Research/Codes/fMRI/rsfMRIwithandwithoutmotion/Matlabcodes/data/datafromOnur/rawfiles/case004/fMRI_nomoco_nomotion_9_4D.nii.gz'
    unsm_im = nib.load(fmri_fname)
    masker  = NiftiMasker(mask_strategy = 'epi')
    masker.fit(unsm_im)
    mean_img = image.mean_img(unsm_im)
    mask_fname = masker.mask_img_
    plotting.plot_roi(mask_fname,mean_img)
    data1 = masker.fit_transform(unsm_im)
    data2 = masker.inverse_transform(data1)
    A = data2.get_fdata()
    #plotting.plot_img(index_img(data2,slice(0,1)),cut_coords = (10,-55,25))
    #plotting.plot_img(index_img(unsm_im,slice(0,1)),cut_coords=(10,-55,25))
    #data2 = apply_mask(unsm_im,mask_fname)
    #temp = masking.unmask(data1,mask_fname,order='C')
    #A = temp.get_fdata()
    B = unsm_im.get_fdata()
    #np.array_equal(A,B)
    #plotting.plot_img(index_img(temp,slice(0,1)))
    #plotting.plot_img(index_img(unsm_im,slice(0,1)))
    #nib.save(unsm_im, 'test_nomo.nii.gz')
    #nib.save(data2, 'nomotionY_bgremoved_temp.nii.gz')
    
    data3 = np.transpose(A,(3,2,1,0))
    Ximg = numpy4Dtositk(data3,origin_4d,direction_4d,spacing_4d,nv)
    sitk.WriteImage(Ximg,outFile)  

#path = '/home/ch208071/Research/Codes/fMRI/rsfMRIwithandwithoutmotion/Matlabcodes/SLR_IRLS/MainCodes_modelincludesmotion/newformulation/experiments/bgmaskrem/filesforsignalclean/'
##fname = '/home/ch208071/Research/Codes/fMRI/rsfMRIwithandwithoutmotion/Matlabcodes/SLR_IRLS/MainCodes_modelincludesmotion/newformulation/experiments/bgmaskrem/filesforsignalclean/*.nii.gz'

fname = '/home/ch208071/Research/Codes/fMRI/rsfMRIwithandwithoutmotion/Matlabcodes/data/datafromYao/abruptmotion.nii.gz'
path = '/fileserver/fetal/Arvind/fMRI/slice_to_volume_fwdmodel/cameradata/'
#fname = '/fileserver/fetal/Arvind/fMRI/DHCP/test_relationshipbetween_mcflirtandsitkmotionparams/test_5toend.nii.gz'
## define variables path and fname. See above for example (lines 73 and 74).
for files in glob.glob(fname):
    print(files)
    NiftiFile = os.path.basename(files)
    f = os.path.splitext(NiftiFile)[0]
    g= os.path.splitext(f)[0]
    outFile = path + g + '_bgremoved.nii.gz'
    maskoutbg(files,outFile)
    
