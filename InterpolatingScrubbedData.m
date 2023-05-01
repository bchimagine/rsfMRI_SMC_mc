function X =  InterpolatingScrubbedData(b,A,At,Mk,na,nb,nsl,n12,ft,mu,p,eta,ts1,ts2,ind_bg,tolerance,maxIter)
%%
Atb = At(b);
X = repmat(mean(Atb,2),[1,nb]);

X_bgrem = X;
X_bgrem(ind_bg,:)=[];

Atb = sparse(Atb);
Atbcell = cell(1,nsl);
for j=1:nsl
    Atbcell{j} = Atb((j-1)*n12+1:j*n12,:);
end
clear Atb
%%
Ir = speye(ft);
n=1;
%blk = na;
blk = size(X_bgrem,1);
%%
%%% Define the optimization parameters.
%p=0.1; %% p for schatten p-norm
q = 1-(p/2);
d = [0:ft-1]';
k = nb-ft+1;
epsilon=1;

cost = [];
term1data= [];
term2reg =[];

for irlsiter=1:maxIter
    %irlsiter
     X_prev= X;
    %%
   %tic;
    for i=1:blk
        Xt{i} = X_bgrem((i-1)*n+1:i*n,:).';
    end
    %toc;
    
    % Weight update
     tic;
    gramR  = zeros(ft);
    parfor l = 1:blk
        HankX = (im2colstep(Xt{l},[ft,1],[1,1]));
        gramR = gramR + (HankX*HankX');
    end
    clear l HankX Xt
    [V,S] = eig(gramR+epsilon*Ir);
    s = abs(diag(S));
    %figure(2);plot(s.^(0.5));title('s values');drawnow;
    clear S gramR
    alpha = s.^(-q/2);
    Q = V*diag(alpha);
    toc;
    %% Precomputing
    % Sum over all ft, Tqi'*Tqi; Tqi is toeplitz matrix formed from qi
    %     Sp = sparse(nb,nb);
    Sp = spalloc(nb,nb,2*ft);
    fcn = @plus;
    tic;
    parfor i1=1:ft
        vrep = repmat(Q(:,i1),[1,k]);
        tempdiag = spdiags(vrep.',d,k,nb);
        Sp= fcn(Sp,(tempdiag.'*tempdiag));
        %         Sp = Sp + (tempdiag.'*tempdiag);
    end
    Sp = mu*Sp;
     toc;
     tic;
    parfor i2 = 1:nsl
        %          X((i2-1)*n12+1:i2*n12,:) = Atb((i2-1)*n12+1:i2*n12,:)/(Sp+Mk{i2});
        Xtemp(:,:,i2) = full(Atbcell{i2}/(Sp+Mk{i2}));
    end
      X = reshape(permute(Xtemp,[1,3,2]),na,nb);
     toc;
   
    clear Xtemp
    X = X.*(X>0);
    
    term1 = b-A(X);
    if p == 0
        schatten = 0.5*sum(log(abs(s-epsilon)));
    else
        schatten = sum(abs((s-epsilon)).^(p/2));
    end

    epsilon_f = epsilon;    
    epsilon = epsilon/eta;
    
    T1 = sum(term1(:).^2);
    T2 = mu*schatten;
    thiscost =  T1 + T2;
    cost = [cost,thiscost];
    
   
    term1data = [term1data,T1];
    term2reg = [term2reg, T2];
    
   figure(3);subplot(1,3,1);plot(cost);title('cost');drawnow;
   subplot(1,3,2);plot(term1data);title('data cons');drawnow;
   subplot(1,3,3);plot(term2reg);title('schatten ');drawnow;
    
    figure(4); plot(X(ts1,:)); drawnow;
    figure(5);plot(X(ts2,:));drawnow;
    
    norm_er = (norm(X-X_prev,'fro')/norm(X_prev,'fro'))
    if  norm_er <=tolerance
        break;
    end
    X_bgrem = X;
    X_bgrem(ind_bg,:)=[];
end
