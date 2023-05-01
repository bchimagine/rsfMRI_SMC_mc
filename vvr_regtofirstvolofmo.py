#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:43:52 2020

@author: ch208071
"""
import numpy as np
import SimpleITK as sitk
import time
#from IPython.display import clear_output
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
#from joblib.externals.loky import set_loky_pickler

#
def convert_quaternions_to_eulerangles(motion_params):
    # intrinsic euler angles from quaternions
    # the rotaion occurs in the order ZYX
    sz = motion_params.shape[0]
    params_euler = np.zeros([sz,6])
    for i in range(sz):
        params = motion_params[i,:]
        q1, q2, q3 = params[0:3]
        tx,ty,tz =  params[3:6]
        q0 = np.sqrt(1 - q1**2 - q2**2 - q3**2)
        
        # for now not worries about order (should rz and rx be interchanged?, for the purposes of computing fd, this will work.)
        # angles in radians
        rz = np.arctan2(2*(q0*q1+q2*q3),(1-2*(q1**2+q2**2)))
        ry = np.arcsin(2*(q0*q2 - q3*q1))
        rx = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))
        
        params_euler[i,:] = [rx,ry,rz,tx,ty,tz]
        
    return params_euler    
        
        
        #print("Quarterion conversion: {}".format([rx, ry, rz]))



def Insert2dSliceinto3Dvolume(I1,I2,j):
    
    return sitk.Paste(I1,I2,I2.GetSize(),destinationIndex=[0,0,j])

def numpy4Dtositk(X_np,origin,direction,spacing,nv):
    
    Xtemp = []
    
    for k in range(0,nv):
        Xtemp.append(sitk.GetImageFromArray(X_np[k,:,:,:]))
        
    Y = sitk.JoinSeries(Xtemp)
    Y.SetOrigin(origin)
    Y.SetSpacing(spacing)
    Y.SetDirection(direction) 
    
    return Y

def get_center(img):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                              int(np.ceil(height/2)),
                                              int(np.ceil(depth/2))))


def resample(X_numpy,tfm,interpolator,origin,spacing,direction,default_value,sz):
    X = sitk.GetImageFromArray(X_numpy)
    X.SetOrigin(origin)
    X.SetSpacing(spacing)
    X.SetDirection(direction)
    R = sitk.ResampleImageFilter()
    R.SetTransform(tfm);
    R.SetOutputOrigin(origin)
    R.SetOutputDirection(direction)
    R.SetOutputSpacing(spacing)
    R.SetDefaultPixelValue(default_value)
    R.SetInterpolator(interpolator)
    R.SetSize(sz)
    return sitk.GetArrayFromImage(R.Execute(X))

#def start_plot():
#     global metric_values
#    
#     metric_values = []
## # Callback invoked when the EndEvent happens, do cleanup of data and figure.
#def end_plot():
#     global metric_values
#    
#     del metric_values
##     # Close figure, we don't want to get a duplicate of the plot later on.
#     plt.close()
#    
#def plot_values(registration_method):
#    global metric_values
#    
#    metric_values.append(registration_method.GetMetricValue())                                       
#    # Clear the output area (wait=True, to reduce flickering), and plot current data
#    clear_output(wait=True)
#    # Plot the similarity metric values
#    plt.plot(metric_values, 'r')
#    plt.xlabel('Iteration Number',fontsize=12)
#    plt.ylabel('Metric Value',fontsize=12)
#    plt.show()


def registerTwoVolumes_withmask(moving,fixed,mask):
    initial_transform =  sitk.Euler3DTransform(sitk.CenteredTransformInitializer(fixed, 
                                                      moving, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY))
    initial_transform.SetFixedParameters([0,0,0,0])
    initial_transform.SetComputeZYX(True)
    registration_method= sitk.ImageRegistrationMethod()

# MI based registration
#    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
#    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
#    registration_method.SetMetricSamplingPercentage(1)
#    registration_method.SetInterpolator(sitk.sitkLinear)
#    registration_method.SetOptimizerAsGradientDescent(learningRate = 0.1,numberOfIterations=100, convergenceMinimumValue=1e-12, convergenceWindowSize=10,estimateLearningRate = registration_method.EachIteration)
#    registration_method.SetOptimizerScalesFromPhysicalShift()
#    registration_method.SetInitialTransform(initial_transform, inPlace=False)
#    transform = registration_method.Execute(fixed, moving)
    
#correlation based registration
    registration_method.SetMetricAsCorrelation()
    registration_method.SetInterpolator(sitk.sitkLinear)
    #registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate = 0.1,numberOfIterations = 35,convergenceMinimumValue = 1e-7 )      
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate = 0.05,numberOfIterations = 50,convergenceMinimumValue = 1e-7 )  
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetMetricFixedMask(mask) # fixed mask
    #registration_method.SetMetricMovingMask(mask) # fixed mask
    transform = registration_method.Execute(fixed, moving)

    return transform

def registerTwoVolumes(moving,fixed):
    initial_transform =  sitk.Euler3DTransform(sitk.CenteredTransformInitializer(fixed, 
                                                      moving, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY))
    
    
#    initial_transform =  sitk.VersorRigid3DTransform(sitk.CenteredTransformInitializer(fixed, 
#                                                      moving, 
#                                                      sitk.VersorRigid3DTransform(), 
#                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY))
##    initial_transform.SetFixedParameters([0,0,0,0])
    initial_transform.SetComputeZYX(True)
    #initial_transform = sitk.Euler3DTransform()
    registration_method= sitk.ImageRegistrationMethod()

# MI based registration
#    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
#    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
#    registration_method.SetMetricSamplingPercentage(1)
#    registration_method.SetInterpolator(sitk.sitkLinear)
#    registration_method.SetOptimizerAsGradientDescent(learningRate = 0.1,numberOfIterations=100, convergenceMinimumValue=1e-12, convergenceWindowSize=10,estimateLearningRate = registration_method.EachIteration)
#    registration_method.SetOptimizerScalesFromPhysicalShift()
#    registration_method.SetInitialTransform(initial_transform, inPlace=False)
#    transform = registration_method.Execute(fixed, moving)
    
#correlation based registration
    registration_method.SetMetricAsCorrelation()
    registration_method.SetInterpolator(sitk.sitkLinear)
    #registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate = 0.1,numberOfIterations = 35,convergenceMinimumValue = 1e-7 )      
    #registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate = 0.05,numberOfIterations = 100,convergenceMinimumValue = 1e-7 ) 
    #registration_method.SetOptimizerAsGradientDescent(learningRate = 1e-3,numberOfIterations=50, convergenceMinimumValue=1e-12, convergenceWindowSize=10,estimateLearningRate = registration_method.EachIteration)
    #registration_method.SetOptimizerAsGradientDescent(learningRate = 1e-7,numberOfIterations=50, convergenceMinimumValue=1e-8, convergenceWindowSize=10)

   
    #registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate = 0.01,numberOfIterations =50,convergenceMinimumValue = 1e-8 ) 
    registration_method.SetOptimizerAsPowell(numberOfIterations = 100, maximumLineIterations=100,stepLength=1,stepTolerance=1e-6,valueTolerance=1e-6)
    #registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=500,maximumNumberOfCorrections=5,  maximumNumberOfFunctionEvaluations=2000, costFunctionConvergenceFactor=1e+7)
    
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
#    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [2,1])
#    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1,0])
#    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()    
    
#    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
#    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
#    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))
    
    #registration_method.SetMetricFixedMask(mask) # fixed mask
    
    transform = registration_method.Execute(fixed, moving)
    
#    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
#    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))    

    return transform

def funcForRegImagesInParallel(I,fixed,origin,spacing,direction,var):
    moving = sitk.GetImageFromArray(I)
    moving.SetOrigin(origin)
    moving.SetSpacing(spacing)
    moving.SetDirection(direction)
    tf =  registerTwoVolumes(moving,fixed)
    #tfm.append(tf.GetParameters())
    Im = sitk.GetArrayFromImage(sitk.Resample(moving,fixed,tf,sitk.sitkLanczosWindowedSinc,0.0,moving.GetPixelID()))
    
    var[0] = np.asarray(tf.GetParameters())
    var[1] = Im
    return var


def RegisterAndSaveUsingJoblib(Y):
    
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
    var = {}
    #nv=16
    #set_loky_pickler('pickle')
    #Y= Parallel(n_jobs = -1,prefer="threads",verbose=1)(delayed(funcForRegImagesInParallel)(I[i-1,:,:,:],fixed,origin,spacing,direction) for i in range(2,nv+1))
    Y= Parallel(n_jobs = -1,verbose=0)(delayed(funcForRegImagesInParallel)(I[i-1,:,:,:],fixed,origin,spacing,direction,var) for i in range(2,nv+1))
    #print(Y)
    len_Y = len(Y)
    tfm =[]
    tfm = np.asarray([Y[i][0] for i in range(len_Y)])
    X_np = np.asarray([Y[i][1] for i in range(len_Y)])
    Final[1:nv,:,:,:] = X_np
    Ximg = numpy4Dtositk(Final,origin_4d,direction_4d,spacing_4d,nv)
     
    res ={}
    res[0] = Ximg
    res[1] = tfm
    
    
    return res

#    for i in range(2,nv+1):
#        #print(i)
#        moving = sitk.GetImageFromArray(I[i-1,:,:,:])
#        moving.SetOrigin(origin)
#        moving.SetSpacing(spacing)
#        moving.SetDirection(direction)
#        tf =  registerTwoVolumes(moving,fixed)
#        tfm.append(tf.GetParameters())
#        Im = sitk.GetArrayFromImage(sitk.Resample(moving,fixed,tf,sitk.sitkLanczosWindowedSinc,0.0,moving.GetPixelID())) 
#        Final[i-1,:,:,:]=Im
#    
#    #return tfm, Final    
#    params_itk = np.asarray(tfm)
#    #np.insert(params_itk,0,np.zeros([1,6]),axis=0)
#    Ximg = numpy4Dtositk(Final,origin_4d,direction_4d,spacing_4d,nv)
#    
#    return Ximg,params_itk
    
    


def RegisterAndSave(Y,ref_vol_number):
    
    n1,n2,nsl,nv = Y.GetSize()
    origin_4d = Y.GetOrigin()
    spacing_4d = Y.GetSpacing()
    direction_4d = Y.GetDirection()
    origin = origin_4d[0:3]
    spacing = spacing_4d[0:3]
    direction = direction_4d[0:3] + direction_4d[4:7] + direction_4d[8:11] # concatenating tuples
    
    I = sitk.GetArrayFromImage(Y)
    #fixed_im = I[0,:,:,:]
    fixed_im = I[ref_vol_number,:,:,:]   # ref volume should be passed as parameter
    fixed = sitk.GetImageFromArray(fixed_im)
    fixed.SetOrigin(origin)
    fixed.SetSpacing(spacing)
    fixed.SetDirection(direction)

    Final = np.zeros([nv,nsl,n2,n1])
    #Final[ref_vol_number,:,:,:] =  fixed_im
    tfm = []
    #nv = 20
    #for i in range(1,nv+1):
    for i in range(nv):
        #print(i)
        #moving = sitk.GetImageFromArray(I[i-1,:,:,:])
        moving = sitk.GetImageFromArray(I[i,:,:,:])
        moving.SetOrigin(origin)
        moving.SetSpacing(spacing)
        moving.SetDirection(direction)
        if i == ref_vol_number:
            Final[i,:,:,:] = fixed_im
            tfm.append((0,0,0,0,0,0))
        else:
            tf =  registerTwoVolumes(moving,fixed)
            tfm.append(tf.GetParameters())
            Im = sitk.GetArrayFromImage(sitk.Resample(moving,fixed,tf,sitk.sitkBSpline,0.0,moving.GetPixelID())) 
        #Final[i-1,:,:,:]=Im
            Final[i,:,:,:]=Im
    
    #return tfm, Final    
    params_itk = np.asarray(tfm)
    #np.insert(params_itk,0,np.zeros([1,6]),axis=0)
    Ximg = numpy4Dtositk(Final,origin_4d,direction_4d,spacing_4d,nv)
    
    return Ximg,params_itk

def RegisterAndSave_withmask(Y,mask):
    
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
        tf =  registerTwoVolumes_withmask(moving,fixed,mask)
        tfm.append(tf.GetParameters())
        Im = sitk.GetArrayFromImage(sitk.Resample(moving,fixed,tf,sitk.sitkBSpline,0.0,moving.GetPixelID())) 
        Final[i-1,:,:,:]=Im
    
    #return tfm, Final    
    params_itk = np.asarray(tfm)
    #np.insert(params_itk,0,np.zeros([1,6]),axis=0)
    Ximg = numpy4Dtositk(Final,origin_4d,direction_4d,spacing_4d,nv)
    
    return Ximg,params_itk
    #sitk.WriteImage(Ximg,Final_fname)  
    #np.savetxt(params_txtfile,params_itk) 
    #sitk.WriteImage(Ximg,'nomotion_Y_registered.nii.gz')  
    #np.savetxt('ITKparams_nomotion.txt',params_itk) 
    
def ResampleVols(Y,params):
    
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
        print(i)
        moving = sitk.GetImageFromArray(I[i-1,:,:,:])
        moving.SetOrigin(origin)
        moving.SetSpacing(spacing)
        moving.SetDirection(direction)
        tf = sitk.Euler3DTransform()
        tf.SetParameters(params[i-2,:])
        tf.SetComputeZYX(True)
        #tf =  registerTwoVolumes(moving,fixed)
        #tfm.append(tf.GetParameters())
        Im = sitk.GetArrayFromImage(sitk.Resample(moving,fixed,tf,sitk.sitkBSpline,0.0,moving.GetPixelID())) 
        Final[i-1,:,:,:]=Im
    
    #return tfm, Final    
    #params_itk = np.asarray(tfm)
    #np.insert(params_itk,0,np.zeros([1,6]),axis=0)
    Ximg = numpy4Dtositk(Final,origin_4d,direction_4d,spacing_4d,nv)
    sitk.WriteImage(Ximg,'/fileserver/fetal/Arvind/fMRI/ABCD/mediummotion/sub-NDARINVC9CUGR6X/ses-baselineYear1Arm1/func/ForAnalysis/rest2/sub-NDARINVC9CUGR6X_ses-baselineYear1Arm1_task-rest_run-02_bold_vvr.nii.gz')
    #return Ximg
    


#Y = sitk.ReadImage('/fileserver/fetal/Arvind/fMRI/ABCD/mediummotion/sub-NDARINVC9CUGR6X/ses-baselineYear1Arm1/func/ForAnalysis/rest1/sub-NDARINVC9CUGR6X_ses-baselineYear1Arm1_task-rest_run-01_bold_vol5toend.nii.gz',sitk.sitkFloat64)
#Yresult = RegisterAndSaveUsingJoblib(Y)
#X_img,params = RegisterAndSave(Y)
# small motion 
#Y = sitk.ReadImage('/home/ch208071/Research/Codes/fMRI/rsfMRIwithandwithoutmotion/bids-dataset/sub-01/ses-01/func/sub-01_ses-01_task-rest_acq-sm10_rec-ori_bold.nii.gz',sitk.sitkFloat64);
#Y = sitk.ReadImage('/home/ch208071/Research/Codes/fMRI/rsfMRIwithandwithoutmotion/bids-dataset/sub-01/ses-01/func/sub-01_ses-01_task-rest_acq-nomo_rec-ori_bold.nii.gz',sitk.sitkFloat64);
#Y = sitk.ReadImage('/home/ch208071/Research/Codes/fMRI/rsfMRIwithandwithoutmotion/bids-dataset/sub-01/ses-01/func/sub-01_ses-01_task-rest_acq-sm20_rec-ori_bold.nii.gz',sitk.sitkFloat64)
#Y = sitk.ReadImage('/home/ch208071/Research/Codes/fMRI/rsfMRIwithandwithoutmotion/Matlabcodes/SLR_IRLS/MainCodes_modelincludesmotion/newformulation/usingmotionparamsfromVVRbasedonsitk/reco_nomo.nii.gz',sitk.sitkFloat64)
#Y = sitk.ReadImage('/home/ch208071/Research/Codes/fMRI/rsfMRIwithandwithoutmotion/Matlabcodes/SLR_IRLS/NilearnfMRIanalysisAndICA/RadiologicallyOrderedDataForICAAndNilearn/Simulation/realdata_nomotion/nomotion_Y.nii.gz',sitk.sitkFloat64)
#n1,n2,nsl,nv = Y.GetSize()
#origin_4d = Y.GetOrigin()
#spacing_4d = Y.GetSpacing()
#direction_4d = Y.GetDirection()
#tfm,Final = Register(Y)
#params_itk = np.asarray(tfm)
#Ximg = numpy4Dtositk(Final,origin_4d,direction_4d,spacing_4d,nv)
#sitk.WriteImage(X_img,'motion_registeredtemp.nii.gz')  
##sitk.WriteImage(Ximg,'ITKcorrregresult_corisiso_sm20_SG.nii.gz')  
#np.savetxt('params_temp.txt',params)

    