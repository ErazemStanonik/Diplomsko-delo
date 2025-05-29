function [W, b] = STM(X, Y,solver, C, epsilon, maxIt)
    % This function trains Support Tensor Machine (STM) using Alternating
    % optimization procedure.
    %
    % X contains input samples. It is a (d+1)-way tensor. If X1 is 3-way tensor 
    % and X contains 3 samples, then it is a 4-way tensor.
    %
    % Y is a vector of class labels.
    %
    % C is regularization parameter.
    addpath('utils');
    
    sizes = X.size;
    d = ndims(X) - 1;  % each sample is a d-way tensor
    m = length(Y);          % we have m samples
    
    assert(sizes(d+1) == m, "Error: Number of samples in X is not the same as number of samples in Y");
    
    % initialize w(k) for k=1:d using cell arrays
    w = cell(1,d);
    for k = 1:d
        w{k} = rand(sizes(k),1);
    end
    
    err = 1;
    it = 0;
    
    % now we do alternating optimization procedure
    while err > epsilon && it < maxIt
        
        err = 0;
    
        for j = 1:d
            dim = size(w{j},1);       % so we know the dimensions of w_j
            % initialize x_i
            x_i = zeros(m,dim);
            w_other = w(setdiff(1:d,j));        % cell with all the w_k except the j-th
            % we obtain parameters beta ...
            beta = prod(cellfun(@(w_k) norm(w_k, 'fro').^2, w_other));
            % ... and x_i
            for i = 1:m
                idx = repmat({':'}, 1, d);
                idx{d + 1} = i;
                x_i(i,:) = calculate_xi(X(idx{:}), w, d, j);
            end
            
            if strcmp(solver, 'fitcsvm')
                % fitcsvm is faster
                w_j_SVM = fitcsvm(x_i,Y,'BoxConstraint',C/beta);
                w_j = w_j_SVM.Beta;
                b = w_j_SVM.Bias;
            elseif strcmp(solver, 'cvx')
                % now we optimize for w{j}
                cvx_begin
                    cvx_quiet true
                    variables w_j(dim,1)
                    variable b
                    variable xi(m,1)
        
                    minimize(0.5*beta*sum_square(w_j) + C * sum(xi))
        
                    subject to
                        Y .* (x_i * w_j + b) >= 1 - xi;
                        xi >= 0;
        
                cvx_end
            else
                fprintf('Solver should be either "fitcsvm" or "cvx" and not "%s".\n', solver);
                return;
            end

            % update error and ...
            err = err + norm(w{j}-w_j, 'fro');
    
            % ... update w
            w{j} = w_j;
        end
        it = it+1;
    end
    W = tensor(toTensor(w));
end