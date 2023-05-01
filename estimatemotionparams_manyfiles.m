%%% Arvind Balachandrasekaran
% Code to simultaneous motion compensated reconstruction and slice time
% correction in rsfMRI

%%% If I get the error "unable to resolve the name py.(module name), add
%%% the directory containing the module to the python path using the
%%% following command:
%%% py.importlib.import_module('vvr_regtofirstvolofmo'), replace vvr_* with
%%% the func name.
clear;close all;clc
set(0,'DefaultFigureWindowStyle','docked');
%%  Add all paths
addpath('./direct-liftandunlift-codes');
addpath('./operators/');
%% create folders to save background volumes and motion params
PWD = pwd;
data_path = '/fileserver/fetal/Arvind/fMRI/slice_to_volume_fwdmodel/DHCP/estimatemotionparams/';
%%% Open the textfile containing the filenames
textfile = strcat(data_path,'dhcp_filenames_formotionparams.txt');
fileId = fopen(textfile);
files = textscan(fileId,'%s');
num_of_files = numel(files{1});
vol_start = 1;
for i = 1:num_of_files
    disp(i)
    fmri_fname = files{1}{i};
    disp(fmri_fname)
    [~,f,~] = fileparts(fmri_fname);
    [~,g,~] = fileparts(f);
    Y = py.SimpleITK.ReadImage(fmri_fname,py.SimpleITK.sitkFloat64);
    %%
    % Estimate motion parameters
    OutputNiftiFile = strcat(data_path,'vvr_mc/',g,'_vvr.nii.gz');
    OutputMotionParams = strcat(data_path,'motionparams/',g,'.txt');
    
    ref_vol_number = 6;
    var = py.vvr_regtofirstvolofmo.RegisterAndSave(Y,int32(ref_vol_number-1));

    Ximg = var{1};
    params_np = var{2};
    py.SimpleITK.WriteImage(Ximg,OutputNiftiFile);
    py.numpy.savetxt(OutputMotionParams,params_np);
end
