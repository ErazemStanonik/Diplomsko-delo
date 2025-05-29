function [W, b] = HrSTM(X,Y,solver,R,C,epsilon,maxIt)
    % This function performs Support Tensor Machine using Alternating
    % optimization procedure. 
    % It proposes W is in CP decomposition format.
    %
    % X contains input samples. It is a (d+1)-way tensor. If X1 is 3-way tensor 
    % and X contains 3 samples, then it is a 4-way tensor.
    %
    % Y is a vector of class labels of shape Y = [l1, l2, ..., lm]'
    %
    % R is rnak of CP decomposition of W
    %
    % C is regularization parameter
    %
    % It returns weight tensor W and bias b
    %
    % WARNING: R should be less than minimum size along d dimensions in
    % sample of X, otherwise B might not be positive-definite and algorithm
    % exits with an error.
    addpath('utils');
    
    sizes = X.size;
    d = ndims(X) - 1;
    m = length(Y);

    assert(sizes(d+1) == m, "Error: Number of samples in X is not the same as number of samples in Y");

    % initialize W as sum of rank-1 tensors
    cores = cell(1,d);
    for k = 1:d
        cores{k} = rand(sizes(k), R);
    end

    err = 1;
    it = 0;

    while it<maxIt && epsilon<err

        cores_old = cores;
        err = 0;

        for j = 1:d
            dim = size(cores{j},1);
            U_j = khatriRao(cores,j,d,R);

            B = U_j' * U_j;
            [V,D] = eig(B);
            % we set the eigenvalues belowe a certain threshold to 0
            % possible improvement for future papers
            % th = 1e-2;
            % D(D<th) = Inf;
            newD = diag(1 ./ sqrt(diag(D)));
            B_sqrt_ = V * newD * V';
            
            Xji_tilde = zeros(m, dim*R);
            for i = 1:m
                idx = repmat({':'}, 1, d);
                idx{d + 1} = i;
                Xji_tilde(i,:) = vec(mode_n_matricization(X(idx{:}),j) * U_j * B_sqrt_);
            end

            try
                if strcmp(solver,'fitcsvm')
                    Uj_SVM = fitcsvm(Xji_tilde,Y,'BoxConstraint',C);
                    Uj_tilde = reshape(Uj_SVM.Beta, [], R);
                    b = Uj_SVM.Bias;
                elseif strcmp(solver,'cvx')
                    cvx_begin
                        cvx_quiet true
                        variable Uj_tilde(dim,R)
                        variable b
                        variable xi(m,1)
        
                        minimize(0.5 * vec(Uj_tilde)' * vec(Uj_tilde) + C * sum(xi))
        
                        subject to 
                            Y .* (Xji_tilde * vec(Uj_tilde) + b) >= 1 - xi;
                            xi >= 0;
                            
                    cvx_end
                else
                    fprintf('Solver should be either "fitcsvm" or "cvx" and not "%s".\n', solver);
                    return;
                end
                
            catch
                fprintf('WARNING: ERROR\nHigherRankSTM could not aproximate W for R = %d.\n', R);
                W = -1;
                b = -1;
                return;
            end

            % update cores{j} and correct error
            cores{j} = Uj_tilde * B_sqrt_;
            err = err + norm(cores{j} - cores_old{j}, 'fro');
        end
        it = it + 1;
    end
    %fprintf('Err = %f, it = %d\n', err, it);
    W = tensor(cpToTensor(cores,ones(R),R,sizes(1:d)));
end