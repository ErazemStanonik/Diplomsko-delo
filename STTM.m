function [tt_cores, b] = STTM(X,Y,n,ranks,d,maxIt,epsilon)
% This function performs Support Tensor Train Machine using  Alternating 
% optimization procedure.
%
% W is in TT format.
% 
% Input samples X is a 1 x m cell containing d 3D core tensors. This is
% done with reshape_cores.m function which transform each sample from 
% tt_tensor to a cell of d core tensors. This should be done beforehand -
% look at testSTTM.m
%
% parameter sample is the original tt_tensor of samples in X. It tells us
% the dimensionality, ranks and shape of samples as a tensor.

m = length(Y);

% initialize W^(k) randomly
tt_cores = cell(1,d);
for i = 1:d
    tt_cores{i} = rand(ranks(i),n(i),ranks(i+1));
end

% cast W in site-1-mixed canonical form. Starting from W^(d) to W^(2). The
% final R factor is absorbed in W^(1)
for k = d:-1:2
    Wk = reshape(tt_cores{k}, ranks(k), n(k)*ranks(k+1));
    [Q,R] = qr(Wk','econ');      % THIN QR decomp.
    tt_cores{k} = reshape(Q', [ranks(k),n(k),ranks(k+1)]);
    tt_cores{k-1} = ttm(tensor(tt_cores{k-1}),R',3).data;
end
% While debugging, Frobenius norm of site-k-mixed canonical form of TT is
% indeed equal to Forbenius norm of k-th TT core

it = 0;
err = 1;

while err > epsilon && it < maxIt
    tt_cores_old = tt_cores;
    for j = 1:d-1
        % compute x_hat
        x_hat = zeros(m, ranks(j)*n(j)*ranks(j+1));
        for i = 1:m
            if j == 1
                R = calculate_part_x_hat(tt_cores,X{i},2,d,'R',n,ranks);
                x_hat(i,:) = vec(squeeze(X{i}{1})*R)'; 
            elseif j == d
                L = calculate_part_x_hat(tt_cores,X{i},1,d-1,'L',n,ranks);
                x_hat(i,:) = vec(L'*X{i}{d})';
            else
                L = calculate_part_x_hat(tt_cores,X{i},1,j-1,'L',n,ranks);
                R = calculate_part_x_hat(tt_cores,X{i},j+1,d,'R',n,ranks);
                ttX = ttm(tensor(X{i}{j}),L',1);
                ttX = ttm(ttX,R',3).data;
                x_hat(i,:) = vec(ttX)';
            end
        end

        % solve for W^(j) and b
        Wk_SVM = fitcsvm(x_hat,Y);
        Wk = reshape(Wk_SVM.Beta, [ranks(j)*n(j),ranks(j+1)]);
        % compute THIN QR decomposition and cast it to site-(j+1)-mixed
        % canonical form
        [Q,R] = qr(Wk,'econ');
        tt_cores{j} = reshape(Q, [ranks(j),n(j),ranks(j+1)]);
        tt_cores{j+1} = ttm(tensor(tt_cores{j+1}),R,1).data;
    end
    for j = d:-1:2
        % compute x_hat
        x_hat = zeros(m, ranks(j)*n(j)*ranks(j+1));
        for i = 1:m
            if j == 1
                R = calculate_part_x_hat(tt_cores,X{i},2,d,'R',n,ranks);
                x_hat(i,:) = vec(squeeze(X{i}{1})*R)'; 
            elseif j == d
                L = calculate_part_x_hat(tt_cores,X{i},1,d-1,'L',n,ranks);
                x_hat(i,:) = vec(L'*X{i}{d})';
            else
                L = calculate_part_x_hat(tt_cores,X{i},1,j-1,'L',n,ranks);
                R = calculate_part_x_hat(tt_cores,X{i},j+1,d,'R',n,ranks);
                ttX = ttm(tensor(X{i}{j}),L',1);
                ttX = ttm(ttX,R',3).data;
                x_hat(i,:) = vec(ttX)';
            end
        end

        % solve for W^(j) and b
        Wk_SVM = fitcsvm(x_hat,Y);
        Wk = reshape(Wk_SVM.Beta, [ranks(j),n(j)*ranks(j+1)]);
        % compute THIN QR decomposition and cast it to site-(j+1)-mixed
        % canonical form
        [Q,R] = qr(Wk','econ');
        tt_cores{j} = reshape(Q', [ranks(j),n(j),ranks(j+1)]);
        tt_cores{j-1} = ttm(tensor(tt_cores{j-1}),R',3).data;
        b = Wk_SVM.Bias;
    end

    % check for error
    err = 0;
    for i =1:d
        err = err + norm(tt_cores{i} - tt_cores_old{i}, 'fro') / norm(tt_cores{i}, 'fro');
    end
    it = it + 1;
    %err
end
err
it