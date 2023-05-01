#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:35:41 2021

@author: ch208071
"""

import numpy as np
import SimpleITK as sitk
from nipype.interfaces import fsl
from nipype.interfaces.fsl import InvWarp
from nilearn.masking import apply_mask
from nilearn import image,plotting, input_data,signal,datasets
from nilearn.image import index_img
from nilearn.input_data import NiftiMasker
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import os
from numpy import linalg as la

def ComputeSNR(Tr,Im):
    #err = la.norm(Tr.flatten())/la.norm(Im.flatten()-Tr.flatten())
    err = la.norm(Im.flatten()-Tr.flatten())/la.norm(Tr.flatten())
    SNR  = 20*np.log10(la.norm(Tr.flatten())/la.norm(Im.flatten()-Tr.flatten()))
    return SNR, err

def GenerateVecFromMatForPlots(M,sz):
    count=0
    v = np.array([])
    for i in range(sz):
        v = np.append(v,M[i,count:sz])
        count = count+1
    
    return v
        

def plot2DHistogram(x,y,Bins):
#    H, xedges, yedges = np.histogram2d(x, y, bins=Bins)
#    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

#    plt.clf()
#    #fig2 = plt.figure()
#    plt.imshow(H.T, extent=extent, origin='lower')
#    plt.xlabel('x')
#    plt.ylabel('y')
#    cbar = plt.colorbar()
#    cbar.ax.set_ylabel('Counts')
#    plt.show()

    #fig3 = plt.figure()
    vmin = 0 
    vmax= 1000
    plt.hist2d(x, y, bins=Bins,range = [[-1,1],[-1,1]])
    #plt.xlabel('x')
    #plt.ylabel('y')
    plt.clim(vmin,vmax)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')

def ModifyAffine(fmri1,fmri2,savepath):
    ## copying the affine of fmri2 to fmri1
    I1 = nib.load(fmri1)
    I2 = nib.load(fmri2)
    I1_mod = nib.Nifti1Image(I1.get_data(), I2.affine, I2.header)
    nib.save(I1_mod,savepath)

def numpy4Dtositk(X_np,origin,direction,spacing,nv):
    
    Xtemp = []
    
    for k in range(0,nv):
        Xtemp.append(sitk.GetImageFromArray(X_np[k,:,:,:]))
        
    Y = sitk.JoinSeries(Xtemp)
    Y.SetOrigin(origin)
    Y.SetSpacing(spacing)
    Y.SetDirection(direction) 
    
    return Y

def ExtractFirstVolFrom4DfMRItimeseries(fMRI_fname,data_path):
    Ypr = sitk.ReadImage(fMRI_fname,sitk.sitkFloat64)
    origin_4d = Ypr.GetOrigin()
    spacing_4d = Ypr.GetSpacing()
    direction_4d = Ypr.GetDirection()
    
    origin = origin_4d[0:3]
    spacing = spacing_4d[0:3]
    direction = direction_4d[0:3] + direction_4d[4:7] + direction_4d[8:11]
    vx,vy,vz,TR = Ypr.GetSpacing()
    Ypr_np = sitk.GetArrayFromImage(Ypr)
    Yref_img = sitk.GetImageFromArray(Ypr_np[0,:,:,:])   ### change 0 to i if vol i is the reference volume.
    Yref_img.SetOrigin(origin)
    Yref_img.SetSpacing(spacing)
    Yref_img.SetDirection(direction)
    sitk.WriteImage(Yref_img,data_path + 'vol1.nii.gz')
    #return Ypr_np[0,:,:,:]
        

def plotSamplefMRISlice(fmri_fname,mask):
    im = nib.load(fmri_fname)
    mean_img = image.mean_img(im)
    plotting.plot_roi(mask,mean_img)
    
def SignalClean(timeseries,hp_cutoff,lp_cutoff,TR):

    timeseries_clean = signal.clean(timeseries, sessions=None, detrend=False, standardize=True, confounds=None, low_pass=lp_cutoff, high_pass=hp_cutoff, t_r=TR, ensure_finite=True)
    return timeseries_clean
    
def PreProcessfMRIForMelodic(fmri_fname,mask,hp_cutoff,lp_cutoff,TR):
    
    Y = sitk.ReadImage(fmri_fname,sitk.sitkFloat64)
    origin_4d = Y.GetOrigin()
    spacing_4d = Y.GetSpacing()
    direction_4d = Y.GetDirection()
    vx,vy,vz,TR = Y.GetSpacing()
    
    Y_np = sitk.GetArrayFromImage(Y)
    nv,nsl,n2,n1 = Y_np.shape
    #timeseries = apply_mask(fmri_fname,mask)
    
    
    masker  = NiftiMasker(mask_img = mask)
    im = nib.load(fmri_fname)
    timeseries = masker.fit_transform(im)
    #plotSamplefMRISlice(fmri_fname,mask)
    timeseries_clean = SignalClean(timeseries,hp_cutoff,lp_cutoff,TR)
    fmri_processed = masker.inverse_transform(timeseries_clean)
    fmri_np = fmri_processed.get_fdata()
    
    fmri_np = np.transpose(fmri_np,(3,2,1,0))
    fmri_img = numpy4Dtositk(fmri_np,origin_4d,direction_4d,spacing_4d,nv)
    
    return fmri_img


def computecorrdirectly(timeseriesA,timeseriesB,savefnamestr):
    
    corr_matrix = (np.dot(timeseriesA.T, timeseriesB) /
                              timeseriesB.shape[0]
                              )
    display=plotting.plot_matrix(corr_matrix,vmax=1.0,vmin=-1.0)
    display.figure.savefig(savefnamestr)
    
    return corr_matrix


def CoordsFromPowerAtlas():
    
    power = datasets.fetch_coords_power_2011()
    print('Power atlas comes with {0}.'.format(power.keys()))
    
    coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
    print('Stacked power coordinates in array of shape {0}.'.format(coords.shape))
    
    return coords


def registerTwoVolumesUsingITK(moving,fixed,fixed_mask,corr_cost):
    
    initial_transform =  sitk.Euler3DTransform(sitk.CenteredTransformInitializer(fixed, 
                                                      moving, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY))
    
    
    initial_transform.SetFixedParameters([0,0,0,0])
    initial_transform.SetComputeZYX(True)
    #initial_transform = sitk.Transform(3, sitk.sitkIdentity)
    
    
    #initial_transform = sitk.CenteredVersorTransformInitializer(fixed,moving,sitk.VersorRigid3DTransform(),computeRotation = False )
    
    registration_method= sitk.ImageRegistrationMethod()
    
    # MI based registration, optimization params could change with datasets.
    # masks can also be used for moving images.
    if corr_cost is 'MI':
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=128)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(1)
        registration_method.SetInterpolator(sitk.sitkLinear)
        #registration_method.SetOptimizerAsGradientDescent(learningRate = 0.05,numberOfIterations=100, convergenceMinimumValue=1e-12, convergenceWindowSize=10,estimateLearningRate = registration_method.EachIteration)
        registration_method.SetOptimizerAsPowell(numberOfIterations = 100, maximumLineIterations=100,stepLength=1,stepTolerance=1e-6,valueTolerance=1e-6)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        if fixed_mask != []:
            registration_method.SetMetricFixedMask(fixed_mask)
        
        transform = registration_method.Execute(fixed, moving)
        
    else:         
    #correlation based registration
        registration_method.SetMetricAsCorrelation()
        registration_method.SetInterpolator(sitk.sitkLinear)    
        #registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate = 0.05,numberOfIterations = 100,convergenceMinimumValue = 1e-7 )  
        registration_method.SetOptimizerAsPowell(numberOfIterations = 100, maximumLineIterations=100,stepLength=1,stepTolerance=1e-6,valueTolerance=1e-6)
         #registration_method.SetOptimizerAsGradientDescent(learningRate = 0.01,numberOfIterations=200, convergenceMinimumValue=1e-12, convergenceWindowSize=10,estimateLearningRate = registration_method.EachIteration)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        if fixed_mask != []:
            registration_method.SetMetricFixedMask(fixed_mask) # fixed mask
            
        #registration_method.SetMetricMovingMask(mask) # fixed mask
        transform = registration_method.Execute(fixed, moving)
    
#    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
#    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
#    
    return transform

def RegisterTwoVolumesUsingFLIRT(moving,fixed, DOF, inp_aff,path_to_save):
# fixed is treated as reference, # moving is in
    flt = fsl.FLIRT(bins=256,dof=DOF,cost='corratio',cost_func='corratio')
    flt.inputs.in_file = moving
    flt.inputs.reference = fixed
    if inp_aff != []:
         flt.inputs.in_matrix_file = inp_aff
         
    matname = path_to_save + "params"+str(DOF)
    flt.inputs.out_matrix_file = matname
    res = flt.run()
    return matname
    
def RegisterTwoVolumesUsingFNIRT(moving, fixed, reg_model,aff_mat, WarpedFile,  FieldFile,  FieldCoeffFile):
    # Before running fnirt, run flirt and get the affine mat file. This file is used in the initialization for fnirt.
    fnt =  fsl.FNIRT()
    fnt = fsl.FNIRT(affine_file=aff_mat)
   # res = fnt.run(ref_file=tg_fname, in_file=src_fname,field_file = 'warpfield.nii.gz',regularization_lambda = Lambda) 
    res = fnt.run(ref_file=fixed, in_file=moving,field_file = FieldFile, warped_file = WarpedFile, fieldcoeff_file = FieldCoeffFile,regularization_model= reg_model )
    
    #return res

def RegisterSourceToStructural(moving, fixed,path_to_save,reg_option):
    DOF = 6
    params_6 = RegisterTwoVolumesUsingFLIRT(moving,fixed,DOF,[],path_to_save + 'MNItoStruct/SourceToStruct_')
    DOF = 12
    params_12 = RegisterTwoVolumesUsingFLIRT(moving,fixed,DOF,params_6,path_to_save + 'MNItoStruct/SourceToStruct_')
    print('FLIRT done')
    if reg_option == 'nonrigid':
        reg_model =  'membrane_energy'
        WarpedFile = path_to_save + 'MNItoStruct/SourceWarpedToStruct.nii.gz'
        FieldFile = path_to_save + 'MNItoStruct/WarpField_SourceToStruct.nii.gz'
        FieldCoeffFile = path_to_save +  'MNItoStruct/FieldCoeff_SourceToStruct.nii.gz'
        RegisterTwoVolumesUsingFNIRT(moving,fixed,reg_model,params_12,WarpedFile, FieldFile, FieldCoeffFile )
        print('FNIRT done')
        
#def RegisterStructuralToTarget(moving,fixed,path_to_save,reg_option):
#    DOF = 6
#    params_6 = RegisterTwoVolumesUsingFLIRT(moving,fixed,DOF,[],path_to_save + 'StructToTarget_')
#    DOF = 12
#    params_12 = RegisterTwoVolumesUsingFLIRT(moving,fixed,DOF,params_6,path_to_save + 'StructToTarget_')
#    print('FLIRT done')
#    if reg_option == 'nonrigid':
#        reg_model =  'membrane_energy'
#        WarpedFile = path_to_save + 'StructWarpedToTarget.nii.gz'
#        FieldFile = path_to_save + 'WarpField_StructToTarget.nii.gz'
#        FieldCoeffFile = path_to_save +  'FieldCoeff_StructToTarget.nii.gz'
#        RegisterTwoVolumesUsingFNIRT(moving,fixed,reg_model,params_12,WarpedFile, FieldFile, FieldCoeffFile )
#        print('FNIRT done')
#        
def RegisterStructuralToTarget(moving,fixed,path_to_save,reg_option,g):
    DOF = 6
    params_6 = RegisterTwoVolumesUsingFLIRT(moving,fixed,DOF,[],path_to_save + g + '_StructToTarget_')
    DOF = 12
    params_12 = RegisterTwoVolumesUsingFLIRT(moving,fixed,DOF,params_6,path_to_save + g + '_StructToTarget_')
    print('FLIRT done')
    if reg_option == 'nonrigid':
        reg_model =  'membrane_energy'
        WarpedFile = path_to_save + 'StructWarpedToTarget_' + g +  '.nii.gz'
        FieldFile = path_to_save + 'WarpField_StructToTarget_' +  g +  '.nii.gz'
        FieldCoeffFile = path_to_save +  'FieldCoeff_StructToTarget_' + g + '.nii.gz'
        RegisterTwoVolumesUsingFNIRT(moving,fixed,reg_model,params_12,WarpedFile, FieldFile, FieldCoeffFile )
        print('FNIRT done')
           
def RegisterVolsoffMRItoFirstVol(Y,fixed_mask,corr_cost):
    
    n1,n2,nsl,nv = Y.GetSize()
    origin_4d = Y.GetOrigin()
    spacing_4d = Y.GetSpacing()
    direction_4d = Y.GetDirection()
    origin = origin_4d[0:3]
    spacing = spacing_4d[0:3]
    direction = direction_4d[0:3] + direction_4d[4:7] + direction_4d[8:11] # concatenating tuples
    
    I = sitk.GetArrayFromImage(Y)
    fixed_im = I[0,:,:,:]
    fixed = sitk.GetImageFromArray(fixed_im)
    fixed.SetOrigin(origin)
    fixed.SetSpacing(spacing)
    fixed.SetDirection(direction)

    Final = np.zeros([nv,nsl,n2,n1])
    Final[0,:,:,:] =  fixed_im
    tfm = []
    for i in range(2,nv+1):
        #print(i)
        moving = sitk.GetImageFromArray(I[i-1,:,:,:])
        moving.SetOrigin(origin)
        moving.SetSpacing(spacing)
        moving.SetDirection(direction)
        tf = registerTwoVolumesUsingITK(moving,fixed,fixed_mask,corr_cost)
        tfm.append(tf.GetParameters())
        Im = sitk.GetArrayFromImage(sitk.Resample(moving,fixed,tf,sitk.sitkBSpline,0.0,moving.GetPixelID())) 
        Final[i-1,:,:,:]=Im
    
    #return tfm, Final    
    params_itk = np.asarray(tfm)
    #np.insert(params_itk,0,np.zeros([1,6]),axis=0)
    Ximg = numpy4Dtositk(Final,origin_4d,direction_4d,spacing_4d,nv)
    
    return Ximg,params_itk

def ApplyFlirtTransform(moving,fixed,params, out_file):
    applyxfm = fsl.ApplyXFM()
    applyxfm.inputs.in_file = moving
    applyxfm.inputs.reference = fixed
    applyxfm.inputs.out_file = out_file
    applyxfm.inputs.in_matrix_file = params
    applyxfm.inputs.apply_xfm = True
    result = applyxfm.run() 
    
def InvertWarpField(reference,FieldFile,OutFile):
    invwarp = InvWarp()
    invwarp.inputs.warp = FieldFile
    invwarp.inputs.reference = reference
    #invwarp.inputs.output_type = "NIFTI_GZ"
    invwarp.inputs.inverse_warp = OutFile
 #invwarp.cmdline
#'invwarp --out=struct2mni_inverse.nii.gz --ref=anatomical.nii --warp=struct2mni.nii'
    res = invwarp.run()         
            
    
def TransformUsingWarpField(moving, fixed, outFile,FieldFile,Interp):
    aw = fsl.ApplyWarp()
    aw.inputs.in_file = moving
    aw.inputs.ref_file = fixed
   # aw.inputs.field_file = 'my_coefficients_filed.nii' 
    aw.inputs.field_file = FieldFile
    aw.inputs.out_file = outFile
    if Interp != []:
        res = aw.run(interp = Interp) 
    else:
        res = aw.run()
        
def TransformCoordsFromSrcToTg(coords,Isrc, Itg, matname,mapping,FieldFile):
    ## If rigid mapping is used, FieldFile = []
    
    #Here we assume the target image is LAS ordered. Checking condition given only in the source side.
    # needs change if target is not LAS ordered.
    Ysrc = sitk.ReadImage(Isrc,sitk.sitkFloat64)
    M1,M2,M3 = Ysrc.GetSize()
    vox_src_x,vox_src_y,vox_src_z = Ysrc.GetSpacing()
    Msrc = nib.load(Isrc).affine 
    
    Ssrc = np.diag(np.array([vox_src_x,vox_src_y,vox_src_z,1]))
#    if np.linalg.det(Msrc[0:4,0:4]) > 0:
#       ScaledMat = np.diag([1,-1,1,1])     # Here the src image follows a LPS coordinate system. 
#       ScaledMat[1,3] = M2-1
#       Ssrc = np.dot(Ssrc,ScaledMat)
#       
    if np.linalg.det(Msrc[0:4,0:4]) > 0:
        ScaledMat = np.diag([-1,1,1,1])     # Here the src image follows a RAS coordinate system. 
        ScaledMat[0,3] = M1-1
        Ssrc = np.dot(Ssrc,ScaledMat)
    
#    Rx =  np.diag(np.array([1,-1,-1,1]))
#    Ry =  np.diag(np.array([-1,1,-1,1]))
#
#    R = np.dot(Ry,Rx)
#    if np.linalg.det(Msrc[0:4,0:4]) > 0:
#        ScaledMat = np.diag([1,-1,1,1])     # Here the src image follows a RAS coordinate system. 
#        #ScaledMat[0,3] = M1-1
#        ScaledMat[1,3] = M2-1
#        Ssrc = np.linalg.multi_dot([Ssrc,ScaledMat,Rx])
#       
       
    Ytg = sitk.ReadImage(Itg,sitk.sitkFloat64)
    N1,N2,N3 = Ytg.GetSize()
    vox_tg_x,vox_tg_y,vox_tg_z = Ytg.GetSpacing()
    Mtg = nib.load(Itg).affine
    
    Stg = np.diag(np.array([vox_tg_x,vox_tg_y,vox_tg_z,1]))
    
    aff_mat= np.loadtxt(matname)
    
    if mapping == 'rigid':
        Tfm_Mat = np.linalg.multi_dot([Mtg,np.linalg.inv(Stg),aff_mat,Ssrc,np.linalg.inv(Msrc)])
    
        coords_tg = []
        for sx,sy,sz in coords:
            n = image.coord_transform(sx,sy,sz,Tfm_Mat)
            #n = n.astype(int)
            n = (n[0],n[1],n[2])
            coords_tg.append(n)
            
    elif mapping == 'nonrigid':
        Tsrc =  np.dot(Ssrc,np.linalg.inv(Msrc))
        T_temp =  np.linalg.multi_dot([np.linalg.inv(Stg),aff_mat,Ssrc,np.linalg.inv(Msrc)])
        Ttg = np.dot(Mtg,np.linalg.inv(Stg))
    
        #img = nib.load('warpfield.nii.gz')
        img = nib.load(FieldFile)
        warpField = img.get_fdata() # numpy
        wx = warpField[:,:,:,0]
        wy = warpField[:,:,:,1]
        wz = warpField[:,:,:,2]
        # Transform the seed points to the functional space.
        coords_tg = []
        for sx,sy,sz in coords:
            
            n = np.asarray(image.coord_transform(sx,sy,sz,Tsrc))   # coord in source in FSL-world coord space
            nvox = np.asarray(image.coord_transform(sx,sy,sz,T_temp))   # coord in func (target) in voxel space
            n[0] = n[0] - wx[int(np.floor(nvox[0])),int(np.floor(nvox[1])),int(np.floor(nvox[2]))]
            n[1] = n[1] - wy[int(np.floor(nvox[0])),int(np.floor(nvox[1])),int(np.floor(nvox[2]))]
            n[2] = n[2] - wz[int(np.floor(nvox[0])),int(np.floor(nvox[1])),int(np.floor(nvox[2]))]
            n = image.coord_transform(n[0],n[1],n[2],Ttg)
            n = (n[0],n[1],n[2])
            coords_tg.append(n)
            
    else :
        print("error! Please specify mapping")

    return coords_tg


def TransformCoordsFromSrcToTgViaStructural(coords,Isrc,Itg,IT1,matname_SourceToStruct,matname_StructToTarget,FieldFile_SourceToStruct,FieldFile_StructToTarget,mapping):
    ## If rigid mapping is used, FieldFile = []
    
    #Here we assume the target image is LAS ordered. Checking condition given only in the source side.
    # needs change if target is not LAS ordered.
    Ysrc = sitk.ReadImage(Isrc,sitk.sitkFloat64)
    M1,M2,M3 = Ysrc.GetSize()
    vox_src_x,vox_src_y,vox_src_z = Ysrc.GetSpacing()
    Msrc = nib.load(Isrc).affine 
    Ssrc = np.diag(np.array([vox_src_x,vox_src_y,vox_src_z,1]))
#    if np.linalg.det(Msrc[0:4,0:4]) > 0:
#       ScaledMat = np.diag([1,-1,1,1])     # Here the src image follows a LPS coordinate system. 
#       ScaledMat[1,3] = M2-1
#       Ssrc = np.dot(Ssrc,ScaledMat)
#       
    if np.linalg.det(Msrc[0:4,0:4]) > 0:
        ScaledMat = np.diag([-1,1,1,1])     # Here the src image follows a RAS coordinate system. 
        ScaledMat[0,3] = M1-1
        Ssrc = np.dot(Ssrc,ScaledMat)
    
#
#    R = np.dot(Ry,Rx)
#    if np.linalg.det(Msrc[0:4,0:4]) > 0:
#        ScaledMat = np.diag([1,-1,1,1])     # Here the src image follows a RAS coordinate system. 
#        #ScaledMat[0,3] = M1-1
#        ScaledMat[1,3] = M2-1
#        Ssrc = np.linalg.multi_dot([Ssrc,ScaledMat,Rx])
# 
    
    Ytg = sitk.ReadImage(Itg,sitk.sitkFloat64)
    N1,N2,N3 = Ytg.GetSize()
    vox_tg_x,vox_tg_y,vox_tg_z = Ytg.GetSpacing()
    Mtg = nib.load(Itg).affine
    Stg = np.diag(np.array([vox_tg_x,vox_tg_y,vox_tg_z,1]))
    
    YT1 = sitk.ReadImage(IT1,sitk.sitkFloat64)
    S1,S2,S3 = YT1.GetSize()
    vox_T1_x,vox_T1_y,vox_T1_z = YT1.GetSpacing()
    MT1 = nib.load(IT1).affine
    ST1 = np.diag(np.array([vox_T1_x,vox_T1_y,vox_T1_z,1]))
    
    SrctoT1= np.loadtxt(matname_SourceToStruct)    # MNI to T1
    T1toTg= np.loadtxt(matname_StructToTarget)   #T1 to func
    if mapping == 'rigid':
        Tfm_Mat = np.linalg.multi_dot([Mtg,np.linalg.inv(Stg),T1toTg,SrctoT1,Ssrc,np.linalg.inv(Msrc)])
    
        coords_tg = []
        for sx,sy,sz in coords:
            n = image.coord_transform(sx,sy,sz,Tfm_Mat)
            #n = n.astype(int)
            n = (n[0],n[1],n[2])
            coords_tg.append(n)
            
    elif mapping == 'nonrigid':
        warpimg_SrctoT1 = nib.load(FieldFile_SourceToStruct)
        warpField_SrctoT1 = warpimg_SrctoT1.get_fdata() # numpy
        wx_A = warpField_SrctoT1[:,:,:,0]
        wy_A = warpField_SrctoT1[:,:,:,1]
        wz_A = warpField_SrctoT1[:,:,:,2]
        
        warpimg_T1toTg = nib.load(FieldFile_StructToTarget)
        warpField_T1toTg = warpimg_T1toTg.get_fdata() # numpy
        wx_B = warpField_T1toTg[:,:,:,0]
        wy_B = warpField_T1toTg[:,:,:,1]
        wz_B = warpField_T1toTg[:,:,:,2]
        
        Msrc_inv = np.linalg.inv(Msrc)
        ST1_inv = np.linalg.inv(ST1)
        TtoSrcworld =  np.dot(Ssrc,Msrc_inv)
        T_temp_A =  np.linalg.multi_dot([ST1_inv,SrctoT1,Ssrc,Msrc_inv]) # takes  you to T1 voxel coordinate system
    #T_toT1 = np.dot(Isrc,Stg_inv)
    
        Stg_inv = np.linalg.inv(Stg)
   # T_mnitoT1 =  np.linalg.multi_dot([MNItoT1,Sref,Mref_inv])
        T_temp_B =  np.linalg.multi_dot([Stg_inv,T1toTg,SrctoT1,Ssrc,Msrc_inv]) # takes you to fMRI voxel coordinate system
        Ttg = np.dot(Mtg,Stg_inv)
    
        # Transform the seed points to the functional space.
        coords_tg = []
        for sx,sy,sz in coords:
            
            n = np.asarray(image.coord_transform(sx,sy,sz,TtoSrcworld))   # coord in MNI in FSL-world coord space
            nvox = np.asarray(image.coord_transform(sx,sy,sz,T_temp_A))   # coord in T1 (target) in voxel space
            n[0] = n[0] - wx_A[int(np.floor(nvox[0])),int(np.floor(nvox[1])),int(np.floor(nvox[2]))]
            n[1] = n[1] - wy_A[int(np.floor(nvox[0])),int(np.floor(nvox[1])),int(np.floor(nvox[2]))]
            n[2] = n[2] - wz_A[int(np.floor(nvox[0])),int(np.floor(nvox[1])),int(np.floor(nvox[2]))]
            
            nvox = np.asarray(image.coord_transform(sx,sy,sz,T_temp_B))   # coord in func (target) in voxel space
            n[0] = n[0] - wx_B[int(np.floor(nvox[0])),int(np.floor(nvox[1])),int(np.floor(nvox[2]))]
            n[1] = n[1] - wy_B[int(np.floor(nvox[0])),int(np.floor(nvox[1])),int(np.floor(nvox[2]))]
            n[2] = n[2] - wz_B[int(np.floor(nvox[0])),int(np.floor(nvox[1])),int(np.floor(nvox[2]))]
            n = image.coord_transform(n[0],n[1],n[2],Ttg)
            n = (n[0],n[1],n[2])
            coords_tg.append(n)       
    else :
        print("error! Please specify mapping")

    return coords_tg
 
def displayCoords(coords,coords_tg,Isrc,Itg,c):
    
    
    Itemp_src = nib.load(Isrc).get_fdata()
    sh = Itemp_src.shape
    if len(sh) == 4:
        Iref_src = index_img(Isrc,slice(0,1))
    else:
        Iref_src = Isrc
    
    
    Itemp_tg = nib.load(Itg).get_fdata()
    
    sh = Itemp_tg.shape
    if len(sh) == 4:
        Iref_tg = index_img(Itg,slice(0,1))
    else:
        Iref_tg = Itg
        
              
    np_coords = np.array(coords)
    display_src=plotting.plot_anat(Iref_src,display_mode = 'ortho',cut_coords = (np_coords[c,0],np_coords[c,1],np_coords[c,2]))
    display_src.add_markers(marker_coords=[[np_coords[c,0],np_coords[c,1],np_coords[c,2]]], marker_color='g',marker_size=75)
    
    np_coordstg = np.array(coords_tg)
    #Iref = index_img(fmri_ref,slice(0,1))
    display_tg=plotting.plot_epi(Iref_tg,cmap = 'gray',display_mode = 'ortho',cut_coords = (np_coordstg[c,0],np_coordstg[c,1],np_coordstg[c,2]))
    display_tg.add_markers(marker_coords=[[np_coordstg[c,0],np_coordstg[c,1],np_coordstg[c,2]]], marker_color='g',marker_size=75)


def CreateNiftiMasker_temp(fMRI_fname,sm_sigma,r_sph,lp_cutoff,hp_cutoff,TR):
    im = nib.load(fMRI_fname)
    masker  = NiftiMasker(mask_strategy = 'epi')
    masker.fit(im)
    mean_img = image.mean_img(im)
    mask_fname = masker.mask_img_
    plotting.plot_roi(mask_fname,mean_img)

    brain_masker = input_data.NiftiMasker(mask_img = mask_fname,smoothing_fwhm=sm_sigma,
            detrend=True, standardize=True, low_pass=lp_cutoff, high_pass=hp_cutoff, t_r=TR,memory='nilearn_cache', memory_level=1, verbose=0)
    
    return brain_masker

def CreateNiftiMasker(sm_sigma,r_sph,lp_cutoff,hp_cutoff,TR):

    brain_masker = input_data.NiftiMasker(mask_strategy = 'epi',smoothing_fwhm=sm_sigma,
            detrend=True, standardize=True, low_pass=lp_cutoff, high_pass=hp_cutoff, t_r=TR,memory='nilearn_cache', memory_level=1, verbose=0)
    
    return brain_masker

def CreateNiftiSpheresMasker(coords,sm_sigma,r_sph,lp_cutoff,hp_cutoff,TR):
    spheres_masker = input_data.NiftiSpheresMasker(
            seeds=coords,allow_overlap=True, smoothing_fwhm=sm_sigma, radius=r_sph,
            detrend=True, standardize=True, low_pass=lp_cutoff, high_pass=hp_cutoff, t_r=TR,memory='nilearn_cache', memory_level=1, verbose=0)
    
    return spheres_masker

def ComputeAndSaveCorrMatixforSBCA(fmri,seed_timeseries,brain_timeseries,spheres_masker,brain_masker,SaveInPath,csv_fname,thresh_corr,thresh_z,coord_ind,coords_tg,optionForScrubbing):
    
    NiftiFile = os.path.basename(fmri)
    #timeseries =  spheres_masker.fit_transform(fmri,confounds=None)
#    seed_timeseries =  spheres_masker.fit_transform(fmri,confounds=csv_fname)
#    brain_timeseries = brain_masker.fit_transform(fmri,confounds=csv_fname)
    
    tmp = os.path.splitext(NiftiFile)[0]
    NameOfFile = os.path.splitext(tmp)[0]
    corr = (np.dot(brain_timeseries.T, seed_timeseries) /
                              seed_timeseries.shape[0]
                              )
    print("Seed-to-voxel correlation shape: (%s, %s)" %
          corr.shape)
    print("Seed-to-voxel correlation: min = %.3f; max = %.3f" % (
    corr.min(), corr.max()))
    
    corr_fisher_z = np.arctanh(corr)
    print("Seed-to-voxel correlation Fisher-z transformed: min = %.3f; max = %.3f"
          % (corr_fisher_z.min(),
             corr_fisher_z.max()
            )
         )
        
    bg = index_img(fmri,slice(0,1))
    seed_to_voxel_correlations_img = brain_masker.inverse_transform(corr.T)
    seed_to_voxel_correlations_fisher_z_img = brain_masker.inverse_transform(corr_fisher_z.T)
#                  
#    display_corr = plotting.plot_stat_map(seed_to_voxel_correlations_img,bg_img = bg,threshold=thresh_corr, vmax=1,cut_coords=coords_tg[coord_ind],draw_cross = False,annotate=False)
#    display_corr.add_markers(marker_coords=[coords_tg[coord_ind]], marker_color='g',marker_size=50)
#        
#    display_fisherZ = plotting.plot_stat_map(seed_to_voxel_correlations_fisher_z_img,bg_img = bg,threshold=thresh_z, vmax=1,
#                                     cut_coords=coords_tg[coord_ind],draw_cross = False,annotate=False)
#    display_fisherZ.add_markers(marker_coords=[coords_tg[coord_ind]], marker_color='g',marker_size=50)
#    
    
    display_corr = plotting.plot_stat_map(seed_to_voxel_correlations_img,bg_img = bg,threshold='auto', vmax=1,cut_coords=coords_tg[coord_ind],draw_cross = False,annotate=False)
    display_corr.add_markers(marker_coords=[coords_tg[coord_ind]], marker_color='g',marker_size=50)
        
    display_fisherZ = plotting.plot_stat_map(seed_to_voxel_correlations_fisher_z_img,bg_img = bg,threshold='auto', vmax=1,
                                     cut_coords=coords_tg[coord_ind],draw_cross = False,annotate=False)
    display_fisherZ.add_markers(marker_coords=[coords_tg[coord_ind]], marker_color='g',marker_size=50)
    
    if optionForScrubbing == False:
        
        corrmatrix_fname = os.path.join(SaveInPath,NameOfFile + '_corr' +  '.pdf')
        display_corr.savefig(corrmatrix_fname)
        
        zmatrix_fname = os.path.join(SaveInPath,NameOfFile + '_fisherZ' +  '.pdf')
        display_fisherZ.savefig(zmatrix_fname)
        
        fissZ_fname =  os.path.join(SaveInPath,NameOfFile + '_fisherZ' + '.nii' + '.gz')
        seed_to_voxel_correlations_fisher_z_img.to_filename(fissZ_fname)
        
        corr_fname =  os.path.join(SaveInPath,NameOfFile + '_corr' + '.nii' + '.gz')
        seed_to_voxel_correlations_img.to_filename(corr_fname)
        
    else:
        
        corrmatrix_fname = os.path.join(SaveInPath,NameOfFile + '_corr_scr' +  '.pdf')
        display_corr.savefig(corrmatrix_fname)
        
        zmatrix_fname = os.path.join(SaveInPath,NameOfFile + '_fisherZ_scr' +  '.pdf')
        display_fisherZ.savefig(zmatrix_fname)
        
        fissZ_fname =  os.path.join(SaveInPath,NameOfFile + '_fisherZ_scr' + '.nii' + '.gz')
        seed_to_voxel_correlations_fisher_z_img.to_filename(fissZ_fname)
        
        corr_fname =  os.path.join(SaveInPath,NameOfFile + '_corr_scr' + '.nii' + '.gz')
        seed_to_voxel_correlations_img.to_filename(corr_fname)
        
    
    return corr,corr_fisher_z,NameOfFile
     
def ComputeAndSaveCorrMatix(fmri,spheres_masker,SaveInPath,csv_fname):
    
    NiftiFile = os.path.basename(fmri)
    #timeseries =  spheres_masker.fit_transform(fmri,confounds=None)
    timeseries =  spheres_masker.fit_transform(fmri,confounds=csv_fname)
    
    tmp = os.path.splitext(NiftiFile)[0]
    NameOfFile = os.path.splitext(tmp)[0]
    corrmatrix_fname = os.path.join(SaveInPath,NameOfFile + '.pdf')
    corrmatrix = computecorrdirectly(timeseries,timeseries,corrmatrix_fname)
    
    return corrmatrix,NameOfFile  

def ConvertToWorldCoords(coords,fname):
    
    Y = sitk.ReadImage(fname,sitk.sitkFloat64)
    M = nib.load(fname).affine 
    
    coords_tg = []
    for sx,sy,sz in coords:
        n = image.coord_transform(sx,sy,sz,M)
        #n = n.astype(int)
        n = (n[0],n[1],n[2])
        coords_tg.append(n)

    return coords_tg

def ReadVoxelCoordsFromExcelandConvertToWorldCoords(fname,seedpoint_csv,newseedpoint_csv):
    
    cols = [3, 4, 5]
    df = pd.read_excel(seedpoint_csv, usecols=cols,index = False, header = None)
    #df = df.iloc[1:] # removing the first row which contains a string. Always check
    df = df.apply (pd.to_numeric, errors='coerce')
    df = df.dropna()
    coords = df.values   # currently the coordinates are specified as voxel coords. Need to convert them to world coordinates

    coords_w = ConvertToWorldCoords(coords-1,fname)  # coords alsoo start from 1 and not 0.
    #
    #df.set_index(['x','y','z'])
    
    pd.DataFrame(coords_w).to_csv(newseedpoint_csv,index = False, header= None)
    
def ExtractingNuisanceSignals(Ymov,Yref,params,reg_model,fmri_fname,mask,corr_cost,WarpedFile,FieldFile,FieldCoeffFile,Y_WM,Y_CSF,outFile_WM,outFile_CSF, ConfoudsCsv_path,path_to_save):
    
    DOF = 6
    params_6 = RegisterTwoVolumesUsingFLIRT(Ymov,Yref,DOF,[],path_to_save)
    print('FLIRt_6 done')
    DOF = 12
    params_12 = RegisterTwoVolumesUsingFLIRT(Ymov,Yref,DOF,params_6,path_to_save)
    print('FLIRT_12 done')
    
    RegisterTwoVolumesUsingFNIRT(Ymov,Yref,reg_model,params_12,WarpedFile, FieldFile, FieldCoeffFile )
    print('FNIRT done')
    
    TransformUsingWarpField(Y_WM,Yref, outFile_WM, FieldFile,'nn')
    TransformUsingWarpField(Y_CSF,Yref, outFile_CSF, FieldFile,'nn')
    print('Transform using warpfield done')
    
    if mask != []:
        if np.allclose(nib.load(mask).affine,nib.load(outFile_WM).affine) !=  True:
            ModifyAffine(outFile_WM,mask,outFile_WM)
    
        if np.allclose(nib.load(mask).affine,nib.load(outFile_CSF).affine) !=  True:
            ModifyAffine(outFile_CSF,mask,outFile_CSF)
            
    
    WM_maskeddata = apply_mask(fmri_fname,outFile_WM)
    CSF_maskeddata = apply_mask(fmri_fname,outFile_CSF)
    GS_maskeddata = apply_mask(fmri_fname, mask)
    print('Masks genereated in func space')
    
#    if datatype == 'ori' or datatype == 'motion':
#        Y = sitk.ReadImage(fmri_fname,sitk.sitkFloat64)
#        Ximg,params = RegisterVolsoffMRItoFirstVol(Y,mask,corr_cost)
#        params = np.insert(params,0,np.zeros([1,6]),0)
#    else:
#        params = []
    
    WM_mean = WM_maskeddata.mean(1)
    CSF_mean = CSF_maskeddata.mean(1)
    GS_mean = GS_maskeddata.mean(1)
    if params == []:
        conf = np.column_stack((WM_mean,CSF_mean, GS_mean))  
        #conf = np.column_stack((WM_mean))  
    else:
        conf = np.column_stack((params,WM_mean,CSF_mean,GS_mean))   ## adding 6 motion params as regressor
        
    pd.DataFrame(conf).to_csv(ConfoudsCsv_path,index = False, header= None)

def ComputeDice(ref,I):
    Yref = sitk.ReadImage(ref,sitk.sitkFloat64)
    Iref = sitk.GetArrayFromImage(Yref)
    
    YI = sitk.ReadImage(I,sitk.sitkFloat64)
    Im = sitk.GetArrayFromImage(YI)
    
    Intersection = np.logical_and(Iref,Im)
    DSC  = 2.* (Intersection.sum()/(Im.sum() + Iref.sum()))
    return DSC
    
def ComputerSensitivityandSpecificity(ref,I):
    # ref and I are numpy matrice
    
    TP  = np.logical_and(ref,I)
    FN = np.logical_and(ref,np.logical_not(I))
    Dr_sum_sens = TP.sum() + FN.sum()
    if Dr_sum_sens == 0:
        sensitivity = 1
    else:
        sensitivity =  TP.sum()/Dr_sum_sens
    
    TN = np.logical_and(np.logical_not(ref),np.logical_not(I))
    FP = np.logical_and(np.logical_not(ref),I)
    Dr_sum_spec = TN.sum() + FP.sum()
    if Dr_sum_spec == 0:
        specificity = 1
    else:
        specificity = TN.sum()/Dr_sum_spec 
    
    return sensitivity, specificity

    
### Test    
#Yref = '../fetal/1271s1/T2andfMRIreg/mean_reconfetal1271s1_masked.nii.gz'
##Yref = sitk.ReadImage('../fetal/1271/T2andfMRIreg/reconfetal_vol1_masked.nii.gz',sitk.sitkFloat64)
##Ymov = '../fetal/1271s1/T2andfMRIreg/t2_t2_1271s1_rad.nii.gz'
##Ymov_atlas = '../fetal/1271s1/T2andfMRIreg/atlas_t2final_1271s1.nii.gz'
#Ymov = '../fetal/1271s1/T2andfMRIreg/STA35.nii.gz'
#mask = '../fetal/1271s1/T2andfMRIreg/mask_1271s1_47.nii.gz'
#path_to_save = '../fetal/1271s1/T2andfMRIreg/'
#
### Transformt the atlas coordinates to the functional space./
#DOF = 6
#params_6 = RegisterTwoVolumesUsingFLIRT(Ymov,Yref,DOF,[],path_to_save)
#print('FLIRt_6 done')
#DOF = 12
#params_12 = RegisterTwoVolumesUsingFLIRT(Ymov,Yref,DOF,params_6,path_to_save)
#ApplyFlirtTransform(Ymov,Yref,params_12,'../fetal/1271s1/T2andfMRIreg/STAandfunc/STA_LAS_flirted.nii.gz')
#print('FLIRT_12 done')
#
#coords = [(-16,-23,23 )]
#mapping = 'rigid'
#coords_afterflirt = TransformCoordsFromSrcToTg(coords,Ymov, Yref, params_12,mapping,[])
#displayCoords(coords,coords_afterflirt,Ymov,Yref,0)
#print('done')
#
#reg_model =  'membrane_energy'
#WarpedFile = '../fetal/1271s1/T2andfMRIreg/STAandfunc/FinalWarped.nii.gz'
#FieldFile = '../fetal/1271s1/T2andfMRIreg/STAandfunc/WarpField.nii.gz'
#FieldCoeffFile =  '../fetal/1271s1/T2andfMRIreg/STAandfunc/FieldCoeff.nii.gz'
#reg_model =  'membrane_energy'
#RegisterTwoVolumesUsingFNIRT(Ymov,Yref,reg_model,params_12,WarpedFile, FieldFile, FieldCoeffFile )
#print('FNIRT done')
#
#coords = [(-16,-23,23 )]
#mapping = 'nonrigid'
#coords_afterflirt = TransformCoordsFromSrcToTg(coords,Ymov, Yref, params_12,mapping,FieldFile)
#displayCoords(coords,coords_afterflirt,Ymov,Yref,0)
#print('done')



#cols = [3, 4, 5]
#df = pd.read_excel(sp_csv, usecols=cols,index = False, header = None)
##df = df.iloc[1:] # removing the first row which contains a string. Always check
#df = df.apply (pd.to_numeric, errors='coerce')
#df = df.dropna()
#coords = df.values   # currently the coordinates are specified as voxel coords. Need to convert them to world coordinates
#
#fname = '../fetal/1271s1/T2andfMRIreg/STA35_LPS.nii.gz'
#coords_w = ConvertToWorldCoords(coords-1,fname)
#pd.DataFrame(coords_w).to_csv("../fetal/seedpointsinSTA/STAseeds_worldcoords_35.xlsx",index = False, header= None)    