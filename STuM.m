function [W, b] = STuM(X, Y, ranks, C, maxIt, epsilon)
    % This function trains Support Tucker Machine (STuM) using Alternating
    % optimization procedure.
    %
    % X is a tensor class containing M samples as d-way tensors
    %
    % Y is a vector of class labels.
    %
    % ranks is a vector of tucker ranks.
    %
    % C is regularization parameter.
    addpath('utils');
       
    sizes = X.size;
    d = ndims(X)-1;
    m = length(Y);

    assert(sizes(d+1) == m, "Error: Number of samples in X is not the same as number of samples in Y");
    
    % initialize core and factor matrices
    G = tensor(rand(ranks));
    P = cell(1,d);
    for k = 1:d
        P{k} = rand(sizes(k),ranks(k));
    end

    err = 1;
    it = 0;
    
    % now we do alternating optimization procedure
    while err > epsilon && it < maxIt
        
        P_old = P;
        G_old = G;

        % optimize factor matrices P
        for j = 1:d
            
            P_j = calculate_pj(P,j,d);
            Hj = mode_n_matricization(G.data, j) * P_j';

            K = Hj * Hj';
            [V,D] = eig(K);
            K_sqrt_ = V * diag(1 ./ sqrt(diag(D))) * V';

            Xj_tilde = zeros(m, sizes(j)*ranks(j));
            for i = 1:m
                idx = repmat({':'}, 1, d);
                idx{d + 1} = i;
                Xj_tilde(i,:) = vec(mode_n_matricization(X(idx{:}).data,j) * Hj' * K_sqrt_);
            end

            try
                cvx_begin
                    cvx_quiet true
                    variable Pj_tilde(size(P{j}))
                    variable b
                    variable zeta(m,1)
    
                    minimize(0.5 * vec(Pj_tilde)' * vec(Pj_tilde) + C * sum(zeta))
    
                    subject to
                        Y .* (Xj_tilde * vec(Pj_tilde) + b) >= 1 - zeta;
                        zeta >= 0;
                        
                cvx_end
            catch
                fprintf('WARNING: ERROR\nSupport Tucker Machine could not aproximate W for given ranks.\n');
                W = -1;
                b = -1;
                return;
            end

            % update P{j}
            P{j} = Pj_tilde * K_sqrt_;
        end

        % optimize core tensor G
        Px = calculate_pj(P,0,d);   % we don't skip any factor matrix
        Xj = zeros(m, prod(sizes(1:d)));
        for i = 1:m
            idx = repmat({':'}, 1, d);
            idx{d + 1} = i;
            Xj(i,:) = vec(X(idx{:}));
        end
        cvx_begin
            cvx_quiet true;
            variable G1(ranks(1),prod(ranks(2:d)))
            variable b
            variable zeta(m,1)

            minimize(0.5 * (Px*vec(G1))' * (Px*vec(G1)) + C * sum(zeta))

            subject to
                Y .* (Xj * Px*vec(G1) + b) >= 1 - zeta;
                zeta >= 0;
        cvx_end

        G = tensor(reshape(G1, ranks));

        % we have to check for err
        err = norm(G - G_old);
        for j = 1:d
            err = err + norm(P{j}-P_old{j});
        end
        it = it + 1;
    end
    % we reconstruct Tucker decomposition of W back to W
    W = ttm(G,P);
end