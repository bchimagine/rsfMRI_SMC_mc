function EstimateMotionParamsFromMotionData(fname,vol_start,OutputNiftiFile,OutputMotionParams)
%Y_mask = [];
Ymo = py.SimpleITK.ReadImage(fname,py.SimpleITK.sitkFloat64);
% origin_4d = Ymo.GetOrigin();
% spacing_4d = Ymo.GetSpacing();
% direction_4d=  Ymo.GetDirection();
% %I = numpytomatlab(py.SimpleITK.GetArrayFromImage(Ymo));
% I =  permute(double(py.SimpleITK.GetArrayFromImage(Ymo)),[3,4,2,1]);
% 
% [n1,n2,nsl,nvs] = size(I);
% %vol_start = 5; % Adjust this parameter to discard initial frames, vol_start =  No of FramesTodiscard + 1
% nv = nvs-vol_start+1;
% %%
% I = I(:,:,:,vol_start:nvs);
% %Inp =  matlabtonumpy(I);
% Inp =  py.numpy.asarray(permute(I,[4,3,1,2]));
% Ymo =  py.vvr_regtofirstvolofmo.numpy4Dtositk(Inp,origin_4d,direction_4d,spacing_4d,int32(nv));
% %var = py.Estintervolumemotionparams.RegisterVolsoffMRItoFirstVol(Ymo,Y_mask,'corr');
% %tic; 
ref_vol_number = vol_start;
var = py.vvr_regtofirstvolofmo.RegisterAndSave(Ymo,int32(ref_vol_number-1));
%toc;
Ximg = var{1};
params_np = var{2};
py.SimpleITK.WriteImage(Ximg,OutputNiftiFile);
py.numpy.savetxt(OutputMotionParams,params_np);
disp('done');
end