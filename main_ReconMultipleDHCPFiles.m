%%% Arvind Balachandrasekaran
% Code to simultaneous motion compensated reconstruction and slice time
% correction in rsfMRI

%%% If you get the error "unable to resolve the name py.(module name), add
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
data_path = '/fileserver/fetal/Arvind/fMRI/slice_to_volume_fwdmodel/DHCP/';
%%% Open the textfile containing the filenames
%%% This text file has to be populated prior to running this code.
textfile = strcat(data_path,'dhcp_filenames_bahram.txt'); 
fileId = fopen(textfile);
files = textscan(fileId,'%s');
num_of_files = numel(files{1});
thresh_noscrub = 0; % variable indicating no scrubbing
thresh_forscrub = 0.5; % variable indicating volumes with framewise displacement > 0.5 will be marked for scrubbing
for i = 1:num_of_files
    disp(i)
    fmri_fname = files{1}{i};
    [~,f,~] = fileparts(fmri_fname);
    [~,g,~] = fileparts(f);
    current_data_path = strcat(data_path,g,'/');
    if ~exist(current_data_path,'dir')
        cd (data_path)
        mkdir(g)
        cd(PWD)
    end

    cd(current_data_path);
    if ~exist('bgremoved','dir')
        mkdir bgremoved
    end

    if ~exist('motionparams','dir')
        mkdir motionparams
    end
    cd(PWD)

    Y = py.SimpleITK.ReadImage(fmri_fname,py.SimpleITK.sitkFloat64);
    origin_4d = Y.GetOrigin();
    spacing_4d = Y.GetSpacing();
    direction_4d=  Y.GetDirection();
    %Iorig = numpytomatlab(py.SimpleITK.GetArrayFromImage(Y));
    Iorig =  permute(double(py.SimpleITK.GetArrayFromImage(Y)),[3,4,2,1]);
    [n1,n2,nsl,nvs] = size(Iorig);
    opt.vol_start = 6; % Adjust this parameter to discard initial frames, vol_start =  No of FramesTodiscard + 1
    nv = nvs-opt.vol_start+1;
    %Iorig_tr = Iorig(:,:,:,vol_start:nvs);
    %Inp =  matlabtonumpy(I);
    %Inp = py.numpy.asarray(permute(Iorig_tr,[4,3,1,2]));
    %Y =  py.vvr_regtofirstvolofmo.numpy4Dtositk(Inp,origin_4d,direction_4d,spacing_4d,int32(nv));
    opt.nv = nv;
    sms_fac=9; %% if multiple slices are acquired at the same time, sms > 1
    n = nsl/sms_fac;
    opt.sms_fac = sms_fac;
    %opt.slice_acq_order = [1:5:nsl,2:5:nsl,3:5:nsl,4:5:nsl,5:5:nsl]; % Order in which the slices are acquired
    opt.slice_acq_order = [1:5:nsl,3:5:nsl,5:5:nsl,2:5:nsl,4:5:nsl]; % Here 5 is the increment, defined as no of slices/sms_fac
    %%
    % Estimate motion parameters
    OutputNiftiFile = strcat(current_data_path,'motionparams/',g,'_vvr.nii.gz');
    OutputMotionParams = strcat(current_data_path,'motionparams/',g,'.txt');

    if ~exist(OutputMotionParams)
        EstimateMotionParamsFromMotionData(fmri_fname,opt.vol_start,OutputNiftiFile,OutputMotionParams);
    end
    %% Compute framewse displacement
    params_temp = load(OutputMotionParams);
    %params_temp = params_temp(opt.vol_start:nvs,:);
    %params = matlabtonumpy(params);
    %opt.params = py.numpy.asarray(params_temp);   %%% numpy stored in opt
    params = params_temp;
    params_temp(:,1:3) = params_temp(:,1:3)*35;%% displacement = r times theta, where r = 35 mm for DHCP
    params_diff = params_temp(1:end-1,:) - params_temp(2:end,:);
    SWD = sum(abs(params_diff),2); % frame wise displacement
    SWD = [0;SWD];
    figure(1),plot(SWD);drawnow;
    y = thresh_forscrub*ones(1,length(SWD));
    hold on,plot(y);drawnow;
    clear SWD params_temp params_diff y
    %%
    opt.n1 = n1;
    opt.n2 = n2;
    opt.nsl = nsl;
    %% fmri time series without background
    fmri_fname_bgremoved = strcat(current_data_path,'bgremoved/',g,'_bgremoved.nii.gz');
    Y_nobg_img = py.SimpleITK.ReadImage(fmri_fname_bgremoved,py.SimpleITK.sitkFloat64);
    I_nobg = permute(double(py.SimpleITK.GetArrayFromImage(Y_nobg_img)),[3,4,2,1]);
    %I_nobg = numpytomatlab(py.SimpleITK.GetArrayFromImage(Ynobg_img));
    I_nobg = I_nobg(:,:,:,opt.vol_start:nvs);
    X_nobg = reshape(I_nobg,[n1*n2*nsl,nv]);
    opt.ind_bg = ~any(X_nobg,2); % index value of 1 corresponds to background pixel.
    clear I_nobg X_nobg Y_nobg_img
    %% set parameters
    opt.mu = 25; % regularization parameter -  From my experiments, I have observed the algorithm to be robust with changing mu.
    %% For now, don't change the value of the pair (beta, beta_fac). Generate the results and we can decide if we want to try a different pair.
    opt.beta = 7.5;%0.25;%7.5;% initial value of beta
    opt.beta_fac = 1.1;%1.1;%1.2; %1.1;% variable to increment beta every iteration
    %%
    opt.ft = 75;% filter size. Determines the size of Hankel matrix formed at every voxel.
    opt.overall_maxIter = 20;%15; % Maximum number of iterations for the algorithm
    opt.maxIter = 6;% Maximum number of iterations for the x-subproblem
    opt.Njobs = -1;% Uses all cores when the python codes corresponding to the z-subproblem are executed
    opt.timepoint1 = 115075;% For display purposes. time series corresponding to it is displayed
    opt.timepoint2 = 104879;% For display purposes. time series corresponding to it is displayed
    opt.ts_start  = 1; % Starting point for downsampling the reconstructed time series
    opt.tolerance = 1e-4; % Stopping point for X subproblem
    opt.eta = 1.1;% parameter for IRLS algorithm. Usually, eta > 1 and < 1.5
    opt.p = 0.1;% Schatten p-norm value 0 <= p <=1

    %% for DHCP we will work on smaller non-overlapping segments
    opt.I1 = Iorig(:,:,:,opt.vol_start);  % for initialization purposes.
    len_segment = 775; % ~ 5 mins of data.
    shift_len = 776 ; %675; % 675 for overlapping, 776 for no overlapping
    count_seg = 1;

    %i = opt.vol_start;
    no_of_segments = 3;
    X_allsegs = cell(1,no_of_segments); %{1,4} for overlapping

    % na = n1*n2*nsl;
    % nb = nsl*nv/sms_fac;
    % X_cat = zeros(na,nb);
    %vol_ind = i:i+len_segment-1;
    vol_start = opt.vol_start;
    vol_ind = opt.vol_start:opt.vol_start+len_segment-1;
    fname = [current_data_path,'recon_noscrub_betainit',num2str(opt.beta),'_','ft' num2str(opt.ft),'_',g,'.nii.gz'];
    if ~exist(fname)
        while (vol_ind(1)<nvs)
            disp(vol_ind(1))
            disp(vol_ind(end))
            if vol_ind(end) <= nvs
                Iseg = Iorig(:,:,:,vol_ind);
                opt.params = py.numpy.asarray(params(vol_ind,:));
                %opt.params = py.numpy.asarray(params(vol_ind-opt.vol_start+1,:));
            else
                new_ind = vol_ind(1):nvs;
                Iseg = Iorig(:,:,:,new_ind);
                opt.params =  py.numpy.asarray(params(new_ind,:));
                %opt.params =  py.numpy.asarray(params(new_ind-opt.vol_start+1,:));
            end
            Iseg_np = py.numpy.asarray(permute(Iseg,[4,3,1,2]));
            Yimg_seg =py.pyfuncs_sv_parallelized_ss_volumelevel.numpy4Dtositk(Iseg_np,origin_4d,direction_4d,spacing_4d,int32(size(Iseg,4)));
            %% First part -  Reconstruct the time series data without scrubbing. The output of this stage is downsampled so that it has the same length as the input.
            [~,X_img,opt] = reconstruct_timeseries_noscrub(Yimg_seg,opt);
            clear Iseg_np Yimg_seg

            X_allsegs{count_seg} = opt.X;
            %i = i + shift_len-1;
            %vol_ind = i:i+len_segment-1;
            vol_start = vol_start + shift_len-1;
            vol_ind = vol_start:vol_start+len_segment-1;

            count_seg = count_seg+1;
        end

        %%% when there is no overlap
        na = n1*n2*nsl;
        nb = nsl*nv/sms_fac;
        X_recon = zeros(na,nb);

        j=1;
        while(j<=no_of_segments)
            X_recon(:,(j-1)*len_segment*nsl/sms_fac +1 : min(j*len_segment*nsl/sms_fac,nb)) = X_allsegs{j};
            j=j+1;
        end
        clear j

        %     %%% when there is overlap
        %     matrix_type = "recon_seg";
        %     X_cat = join_reconstructedSegments_DHCP(matrix_type,X_allsegs,n1,n2,nsl,nv,sms_fac,len_segment,shift_len,1);
        %     matrix_type = "weight";
        %     W = join_reconstructedSegments_DHCP(matrix_type,X_allsegs,n1,n2,nsl,nv,sms_fac,len_segment,shift_len,1);
        %
        %     na = n1*n2*nsl;
        %     nb = nsl*nv/sms_fac;
        %     X_recon = X_cat./W; %%% overlapping segments are averaged.

        X_recon_sampled =  X_recon(:,opt.ts_start:nsl/sms_fac:nb);
        %fname = strcat(data_path,'recon_noscrub_betainit',num2str(opt.beta),'_',g,'_oseg_averaged.nii.gz');
        %fname = strcat(data_path,'recon_',g,'_oseg_averaged.nii.gz');
        %fname = [current_data_path,'recon_noscrub_betainit',num2str(opt.beta),'_','ft' num2str(opt.ft),'_',g,'.nii.gz'];

        %fname = strcat(data_path,'recon_',g,'_beta0.2_fac1.2_iter20_segnooverlap.nii.gz');

        Isamp = reshape(X_recon_sampled,n1,n2,nsl,nv);
        X_numpy = py.numpy.asarray(permute(Isamp,[4,3,1,2]));
        %X_numpy = matlabtonumpy(Isamp);
        X_img = py.pyfuncs_sv_parallelized_ss_volumelevel.numpy4Dtositk(X_numpy,origin_4d,direction_4d,spacing_4d,int32(nv));
        py.SimpleITK.WriteImage(X_img, fname);
    else
        X_img = py.SimpleITK.ReadImage(fname,py.SimpleITK.sitkFloat64);
    end

%     keyboard
    %fname = strcat(data_path,'recon_noscrub_betai nit',num2str(opt.beta),'_',g,'.nii.gz');
    %py.SimpleITK.WriteImage(X_img, fname);
    %X_img = py.SimpleITK.ReadImage(fname,py.SimpleITK.sitkFloat64);
    %% The downsampled output is the input to the second stage, where volumes with FD > threshold are marked for scrubbing.
    %% The missing data is then interpolated using the matrix completion algorithm.
    %% Interpolate scrubbed data.
    opt.lower_thresh = -1;
    opt.higher_thresh = 2; %% Threshold to remove volumes.
    opt.tolerance = 1e-6;
    %opt.maxIter = 45;
    opt.maxIter = 12;
    opt.thresh_forscrub = thresh_forscrub;

    params_temp = load(OutputMotionParams);
    %params_temp = params_temp(opt.vol_start:nvs,:);
    %params = matlabtonumpy(params);
    %opt.params = py.numpy.asarray(params_temp);   %%% numpy stored in opt
    params_temp = params_temp(opt.vol_start:nvs,:);
    params_temp(:,1:3) = params_temp(:,1:3)*35;%% displacement = r times theta, where r = 35 mm for DHCP
    params_diff = params_temp(1:end-1,:) - params_temp(2:end,:);
    SWD = sum(abs(params_diff),2); % frame wise displacement
    SWD = [0;SWD];
    figure(),plot(SWD);drawnow;
    y = thresh_forscrub*ones(1,length(SWD));
    hold on,plot(y);drawnow;

    opt.SWD = SWD;
    opt.beta = 1; %% Don't change.

%     figure(7),plot(SWD);drawnow;
%     y = thresh_forscrub*ones(1,length(SWD));
%     hold on,plot(y);drawnow;

    [vol_i,~,~] = find(SWD>=thresh_forscrub);
    if ~isempty(vol_i)
        %opt.filt_siz = nv/2;   %% Have set the filter size to length of series/2. If the results are not good, run the algorithm for a different value.
        opt.filt_siz = 1100;
        for count = 1:length(opt.filt_siz)
            opt.ft = opt.filt_siz(count);
            [Xs,vol_rem] = MainFunctionForInterpScrubbedData(X_img,opt);
            Is = reshape(Xs,n1,n2,nsl,nv);
            X_np = py.numpy.asarray(permute(Is,[4,3,1,2]));
            %X_np = matlabtonumpy(Is);
            X_img = py.pyfuncs_sv_parallelized_ss_volumelevel.numpy4Dtositk(X_np,origin_4d,direction_4d,spacing_4d,int32(nv));
            filename   = [current_data_path,'recon_scrub_betainit',num2str(opt.beta),'_','ft' num2str(opt.ft),'_',g,'.nii.gz'] ;
            py.SimpleITK.WriteImage(X_img, filename);
        end
        py.numpy.save(strcat(current_data_path,g,'_vol_rem'),matlabtonumpy(vol_rem));
    end
    disp(strcat(g, '    done'));
end
