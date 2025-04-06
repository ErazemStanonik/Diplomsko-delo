function [W, b] = HigherRankSTM(X,Y,R,C,epsilon,maxIt)
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
    
    sizes = size(X);
    d = length(sizes) - 1;
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
            B_sqrt_ = V * diag(1 ./ sqrt(diag(D))) * V';
            
            Xji_tilde = cell(1,m);
            for i = 1:m
                idx = repmat({':'}, 1, d);
                idx{d + 1} = i;
                Xji_tilde{i} = vec(mode_n_matricization(X(idx{:}),j) * U_j * B_sqrt_);
            end

            try
                cvx_begin
                    cvx_quiet true
                    variable Uj_tilde(dim,R)
                    variable b
                    variable zeta(m,1)
    
                    minimize(0.5 * vec(Uj_tilde)' * vec(Uj_tilde) + C * sum(zeta))
    
                    subject to 
                        for i = 1:m
                            Y(i) .* (vec(Uj_tilde)' * vec(Xji_tilde{i}) + b) >= 1 - zeta(i);
                            zeta(i) >= 0;
                        end
                        
                cvx_end
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
    W = cpToTensor(cores,ones(R),R,sizes(1:d));
end