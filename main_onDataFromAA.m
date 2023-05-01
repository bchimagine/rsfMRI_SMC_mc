%%% Arvind Balachandrasekaran
% Code to simultaneous motion compensated reconstruction and slice time
% correction in rsfMRI
clear all; close all;clc
disp('Set paths and make directories');
set(0,'DefaultFigureWindowStyle','docked');
%%  Add all paths
addpath('/fileserver/fetal/fmri/motioncompensation/rsfMRI-motioncompensation');
addpath('./direct-liftandunlift-codes');
addpath('./operators/');
%% create folders to save background volumes and motion params
PWD = pwd;
subj_path = '/fileserver/fetal/fmri/motioncompensation/datafromAA/LT_4934901_26192150';
data_path = strcat(subj_path,'/input/');
files = dir(strcat(data_path,'*.nii.gz'));
bgrm_path = strcat(data_path,'/bgremoved/');
mot_path = strcat(subj_path,'/motionparams/');
out_path = strcat(subj_path,'/output/');
l_f = length(files);

cd(subj_path);
if ~exist(bgrm_path,'dir')
    mkdir(bgrm_path)
end
if ~exist(mot_path,'dir')
    mkdir(mot_path)
end
if ~exist(out_path,'dir')
    mkdir(out_path)
end
cd(PWD)
thresh_noscrub = 0; % variable indicating no scrubbing
thresh_forscrub = 0.5; % variable indicating volumes with framewise displacement > 0.5 will be marked for scrubbing
for i = 1:length(files)
    fmri_fname = strcat(data_path,files(i).name);
    disp(strcat('fMRI input: ',fmri_fname));
    [~,f,~] = fileparts(files(i).name);
    [~,g,~] = fileparts(f);
    disp('SimpleITK ReadImage and GetArray');
    Y = py.SimpleITK.ReadImage(fmri_fname,py.SimpleITK.sitkFloat64);
    origin_4d = Y.GetOrigin();
    spacing_4d = Y.GetSpacing();
    direction_4d=  Y.GetDirection();
    Iorig = numpytomatlab(py.SimpleITK.GetArrayFromImage(Y));
    [n1,n2,nsl,nvs] = size(Iorig);
    opt.vol_start = 1; % Adjust this parameter to discard initial frames, vol_start =  No of FramesTodiscard + 1
    vol_start = opt.vol_start
    nv = nvs-opt.vol_start+1;
    opt.nv = nv;
    sms_fac=1; %% if multiple slices are acquired at the same time, sms > 1
    n = nsl/sms_fac;
    opt.sms_fac = sms_fac;
    opt.slice_acq_order = [1:2:nsl,2:2:nsl]; % Order in which the slices are acquired
    %%
    % Estimate motion parameters
    OutputNiftiFile = strcat(mot_path,g,'_vvr.nii.gz');
    OutputMotionParams = strcat(mot_path,g,'.txt');

    if ~exist(OutputMotionParams)
        disp('EstimateMotionParamsFromMotionData');
        EstimateMotionParamsFromMotionData(fmri_fname,vol_start,OutputNiftiFile,OutputMotionParams);
    end
    %% Compute framewse displacement
    params_temp = load(OutputMotionParams);
    params_temp = [zeros(1,6);params_temp];%% set zeros for first volume (ref)
    starting_index = 1;
    params = zeros(nv,6);
    params(:,1:6) = params_temp(starting_index:nv,1:6);
    params_temp = params;
    params = matlabtonumpy(params);
    
    params_temp(:,1:3) = params_temp(:,1:3)*50;%% displacement = r times theta, where r = 50 mm
    params_diff = params_temp(1:end-1,:) - params_temp(2:end,:);
    SWD = sum(abs(params_diff),2); % frame wise displacement
    SWD = [0;SWD];
    %figure(i),plot(SWD);drawnow;
    y = thresh_noscrub *ones(1,length(SWD));
    %hold on,plot(y);drawnow;
   %%
    opt.params = params;   
    opt.n1 = n1;
    opt.n2 = n2;
    opt.nsl = nsl;  
    %% fmri time series without background
    disp('SimpleITK for bgremoved');
    fmri_fname_bgremoved = strcat(bgrm_path,g,'_bgremoved.nii.gz');
    Ynobg_img = py.SimpleITK.ReadImage(fmri_fname_bgremoved,py.SimpleITK.sitkFloat64);
    I_nobg = numpytomatlab(py.SimpleITK.GetArrayFromImage(Ynobg_img));
    I_nobg = I_nobg(:,:,:,opt.vol_start:nvs);
    X_nobg = reshape(I_nobg,[n1*n2*nsl,nv]);
    opt.ind_bg = ~any(X_nobg,2); % index value of 1 corresponds to background pixel.
    %% set parameters
    opt.mu = 25; % regularization parameter -  From my experiments, I have observed the algorithm to be robust with changing mu. 
    %% For now, don't change the value of the pair (beta, beta_fac). Generate the results and we can decide if we want to try a different pair.
    opt.beta = 7.5;% initial value of beta
    opt.beta_fac = 1.05;% variable to increment beta every iteration
    %%
    opt.ft = 50;% filter size. Determines the size of Hankel matrix formed at every voxel.
    opt.overall_maxIter = 15; % Maximum number of iterations for the algorithm
    opt.maxIter = 6;% Maximum number of iterations for the x-subproblem
    opt.Njobs = -1;% Uses all cores when the python codes corresponding to the z-subproblem are executed
    opt.timepoint1 = 272581;% For display purposes. time series corresponding to it is displayed
    opt.timepoint2 = 270006;% For display purposes. time series corresponding to it is displayed
    opt.ts_start  = 1; % Starting point for downsampling the reconstructed time series
    opt.tolerance = 1e-3; % Stopping point for X subproblem
    opt.eta = 1.1;% parameter for IRLS algorithm. Usually, eta > 1 and < 1.5
    opt.p = 0.1;% Schatten p-norm value 0 <= p <=1
    %% First part -  Reconstruct the time series data without scrubbing. The output of this stage is downsampled so that it has the same length as the input. 
    [~,X_img,opt] = reconstruct_timeseries_noscrub(fmri_fname,opt);
    fname = strcat(out_path,'recon_noscrub_betainit',num2str(opt.beta),'_',g,'.nii.gz');
    disp('Write image');
    py.SimpleITK.WriteImage(X_img, fname);
    %X_img = py.SimpleITK.ReadImage(fname,py.SimpleITK.sitkFloat64);
%% The downsampled output is the input to the second stage, where volumes with FD > threshold are marked for scrubbing.
%% The missing data is then interpolated using the matrix completion algorithm.
    %% Interpolate scrubbed data.
    opt.lower_thresh = -1;
    opt.higher_thresh = 2; %% Threshold to remove volumes.
    opt.tolerance = 1e-6;
    opt.maxIter = 45;   
    opt.thresh_forscrub = thresh_forscrub;
    opt.SWD = SWD;
    opt.beta = 1; %% Don't change. 
    
    %figure(i),plot(SWD);drawnow; 
    y = thresh_forscrub*ones(1,length(SWD));
    %hold on,plot(y);drawnow;

    [vol_i,~,~] = find(SWD>=thresh_forscrub);
    
    if ~isempty(vol_i)
        opt.filt_siz = nv/2;   %% Have set the filter size to length of series/2. If the results are not good, run the algorithm for a different value. 
        disp('InterpScrubbedData loop');
        for count = 1:length(opt.filt_siz)
            opt.ft = opt.filt_siz(count);
            [Xs,vol_rem] = MainFunctionForInterpScrubbedData(X_img,opt);
            Is = reshape(Xs,n1,n2,nsl,nv);
            X_np = matlabtonumpy(Is);
            X_img = py.pyfuncs_sv_parallelized_ss_volumelevel.numpy4Dtositk(X_np,origin_4d,direction_4d,spacing_4d,int32(nv));
            filename   = [out_path,'recon_scrub_betainit',num2str(opt.beta),'_','ft' num2str(opt.ft),'_',g,'.nii.gz'] ;
            py.SimpleITK.WriteImage(X_img, filename);
        end
        disp('Post scrub save');
        py.numpy.save(strcat(out_path,'vol_rem_thr0.5_afterscrubbing'),matlabtonumpy(vol_rem));
    end
    disp(strcat(g, '    done'));
end
