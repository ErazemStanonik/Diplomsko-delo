function [W,b] = LTR(X,Y,R,epsilon,maxIt)
% This function performs Logistic Tensor Regression using Alternating
% optimization procedure. 
% It proposes W is in CP decomposition format.
%
% X contains d-way input samples. It is a (d+1)-way tensor. 
%
% Y is a vector of class labels of shape Y = [l1, l2, ..., lm]'
%
% R is rank of CP decomposition of W
%
% C is regularization parameter
%
% It returns weight tensor W and bias b

sizes = X.size;
d = ndims(X) - 1;
m = length(Y);

% initialize w{k} randomly
cores = cell(1,d);
for k = 1:d
    cores{k} = rand(sizes(k),R);
end

idx = repmat({':'},1,d);

it = 0;
err = 1;

while epsilon < err && it < maxIt
    cores_old = cores;
    % Alternating optimization procedure
    for j = 1:d
        dim = size(cores{j},1);
        U_j = khatriRao(cores,j,d,R);
        Xji_tilde = zeros(m,dim*R);
        for i = 1:m
            Xji_tilde(i,:) = vec(mode_n_matricization(X(idx{:},i),j) * U_j)';
        end
        % now we obtain a LR model
        LR = fitclinear(Xji_tilde,Y,'Learner','logistic', 'Solver','lbfgs');
        cores{j} = reshape(LR.Beta, [dim,R]);
        b = LR.Bias;
    end
    % check for error
    err = 0;
    for j = 1:d
        err = err + norm(cores{j}-cores_old{j}, 'fro') / norm(cores{j}, 'fro');
    end
    it = it + 1;
end
fprintf('Err = %f, it = %d\n', err, it);
disp('-------');
W = tensor(cpToTensor(cores,ones(R),R,sizes(1:d)));