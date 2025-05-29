function [W,b] = r1LR(X,Y,solver,epsilon,maxIt)
% This function trains Rank-1 Logistic Regression (r1LR) using Alternating
% optimization procedure
%
% X contains d-way input samples. It is a (d+1)-way tensor
%
% Y is a vector of class labels
%
%
addpath('utils');

sizes = X.size;
d = ndims(X) - 1;
m = length(Y);

% initialize w(k) for k=1:d randomly
w = cell(1,d);
for k = 1:d
    w{k} = rand(sizes(k),1);
end
    
err = 1;
it = 0;

% begin alternating optimization procedure
while err > epsilon && it < maxIt
    w_old = w;
    for j = 1:d
        dim = size(w{j},1);
        idx = repmat({':'},1,d);
        % calculate x_i
        x_i = zeros(m,dim);
        for i = 1:m
            x_i(i,:) = calculate_xi(X(idx{:},i), w, d, j);
        end
        % now we get weights and bias
        if strcmp(solver,'fitclinear')
            wj_LR = fitclinear(x_i,Y,'Learner','logistic', 'Solver','lbfgs');
            w{j} = wj_LR.Beta;
            b = wj_LR.Bias;
        elseif strcmp(solver,'cvx')
            [wi,b] = myLR(x_i,Y);
            w{j} = wi;
        else
            fprintf('Solver should be either "fitclinear" or "cvx" and not "%s".\n', solver);
            return;
        end
    end
    % chack for error
    err = 0;
    for j = 1:d
        err = err + norm(w{j}-w_old{j}, 'fro') / norm(w{j},'fro');
    end
    it = it + 1;
end
%fprintf('Err = %f it = %d\n', err, it);
W = tensor(toTensor(w));