function B = fwd_model_onZ(Z,interpolator,params,origin,spacing,direction,origin_4d,spacing_4d,direction_4d,n1,n2,nsl,nv)

% Z is a numpy variable.
%Z_np_old = matlabtonumpy(Z);
Z_np = py.numpy.asarray(permute(Z,[4,3,1,2]));

% numpy to simple-ITK object
Z_img = py.pyfuncs_sv_parallelized_ss_volumelevel.numpy4Dtositk(Z_np,origin_4d,direction_4d,spacing_4d,int32(nv));

Zv = py.pyfuncs_sv_parallelized_ss_volumelevel.fwdMotionOperator(Z_img,interpolator,origin,spacing,direction,params,int32(n1),int32(n2),int32(nsl),int32(nv));
%B_old = numpytomatlab(Zv); % returns a 4D matrix
B = permute(double(Zv),[3,4,2,1]); % returns a 4D matrix
end
        