function X_cat = join_reconstructedSegments_DHCP(matrix_type,X_allsegs,n1,n2,nsl,nv,sms_fac,len_segment,shift_len,vol_start)
% matrix type can be recon segments or weight matrix
na = n1*n2*nsl;
nb = nsl*nv/sms_fac;
X_cat = zeros(na,nb);
%vol_start = 1;
count_seg = 1;
vol_ind = vol_start:vol_start+(len_segment*nsl/sms_fac)-1;

while vol_ind(1)< nb
    temp_X = zeros(na,nb);
    if matrix_type == "recon_seg"
        temp_X(:,vol_ind(1):min(vol_ind(end),nb)) = X_allsegs{count_seg};
    else
         temp_X(:,vol_ind(1):min(vol_ind(end),nb)) = ones(size(X_allsegs{count_seg}));
    end
    X_cat = X_cat + temp_X;
    %i = i + (shift_len*nsl/sms_fac)-1;
    vol_start = vol_start + (shift_len-1)*nsl/sms_fac;
    vol_ind = vol_start:vol_start+(len_segment*nsl/sms_fac)-1;
    count_seg = count_seg+1;
    clear temp_X
end

