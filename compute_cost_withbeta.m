function thiscost = compute_cost_withbeta(Z,Y,X,s,epsilon,p,mu,beta,interpolator,params,origin,spacing,direction,origin_4d,spacing_4d,direction_4d,n1,n2,nsl,nv)

B = @(z)fwd_model_onZ(z,interpolator,params,origin,spacing,direction,origin_4d,spacing_4d,direction_4d,n1,n2,nsl,nv);

t1 = B(Z) - Y;
t2 = Z-X;
schatten = sum(abs((s-epsilon)).^(p/2));
thiscost = sum(t1(:).^2) + beta*sum(t2(:).^2)+mu*schatten;
end