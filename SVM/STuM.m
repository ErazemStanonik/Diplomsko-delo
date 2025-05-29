function [W, b] = STuM(X, Y, solver, ranks, C, epsilon, maxIt)
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
    G = tensor(rand(ranks),ranks);
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
            Hj = mode_n_matricization(G, j) * P_j';

            K = Hj * Hj';
            [V,D] = eig(K);
%             th = 5e-3;
%             D(D < th) = Inf;
            K_sqrt_ = V * diag(1 ./ sqrt(diag(D))) * V';
            %K_sqrt_ = pinv(sqrtm(K));

            Xj_tilde = zeros(m, sizes(j)*ranks(j));
            for i = 1:m
                idx = repmat({':'}, 1, d);
                idx{d + 1} = i;
                Xj_tilde(i,:) = vec(mode_n_matricization(X(idx{:}),j) * Hj' * K_sqrt_)';
            end

            try
                if strcmp(solver, 'cvx')
                    cvx_begin
                        cvx_quiet true
                        variable Pj_tilde(size(P{j}))
                        variable b
                        variable xi(m,1)
        
                        minimize(0.5 * vec(Pj_tilde)' * vec(Pj_tilde) + C * sum(xi))
        
                        subject to
                            Y .* (Xj_tilde * vec(Pj_tilde) + b) >= 1 - xi;
                            xi >= 0;
                    cvx_end
                elseif strcmp(solver, 'fitcsvm')
                    % fitcsvm is faster than cvx
                    Pj_SVM = fitcsvm(Xj_tilde,Y,'BoxConstraint',C);
                    Pj_tilde = reshape(Pj_SVM.Beta, sizes(j),[]);
                    b = Pj_SVM.Bias;
                end
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
        X1 = zeros(m, prod(sizes(1:d)));
        for i = 1:m
            idx = repmat({':'}, 1, d);
            idx{d + 1} = i;
            X1(i,:) = vec(X(idx{:}))';
        end
        %if strcmp(solver, 'cvx')

        % after endless hours of testing and debugging it turns out that
        % fitcsvm doesn't give good resultes in case of computing G, so we
        % must use CVX in this step.
            cvx_begin
                cvx_quiet true;
                variable G1(ranks(1),prod(ranks(2:d)))
                variable b
                variable xi(m,1)
    
                minimize(0.5 * (Px*vec(G1))' * (Px*vec(G1)) + C * sum(xi))
    
                subject to
                    Y .* (X1 * Px*vec(G1) + b) >= 1 - xi;
                    xi >= 0;
            cvx_end
%         elseif strcmp(solver, 'fitcsvm')
%             % again here ...
%             G_SVM = fitcsvm(X1,Y,'BoxConstraint',C);
%             G1 = pinv(Px)*G_SVM.Beta;
%             %G1 = Px \ G_SVM.Beta;
%             b = G_SVM.Bias;
%         end

        G = tensor(reshape(full(G1), ranks), ranks);

        % we have to check for err
        err = norm(G - G_old);
        for j = 1:d
            err = err + norm(P{j}-P_old{j},'fro') / norm(P{j},'fro');
        end
        it = it + 1;
    end
    % we reconstruct Tucker decomposition of W back to W
    W = ttm(G,P);
    %fprintf('Err = %f, it = %d\n', err, it);
end