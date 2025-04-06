function [cores, lambda] = CP(X,R,maxIt,epsilon)
    % This function is modification of a function cp3_DCPD.m from
    % https://www.mathworks.com/matlabcentral/fileexchange/72932-cp-decomposition-simple-implementation
    % However it was only suitable for 3-way tensors and I generalized it to a
    % d-way tensors. I looked up to https://arxiv.org/pdf/2112.10855
    %
    % X is an input d-way tensor
    %
    % R is desired CP rank
    %
    % cores is a cell variable containing d matrices
    %
    % lambda is a scaling factor for each component r = 1 ... R

    sz = size(X);
    d = length(sz);
    it = 0;
    err = 1;
    
    lambda = ones(R,1);
    cores = cell(1, d);
    for i = 1:d
        % 1st core is of size size(X,1) x R and so on ...
        cores{i} = rand(sz(i), R);
    end

    while it<maxIt && epsilon<err
        cores_old = cores;

        for j = 1:d
            Sn = ones(R,R);
            for k = setdiff(1:R,j)
                Sn = Sn .* (cores{k}' * cores{k});
            end

            Zn = khatriRao(cores,j,d,R);
            Xn = mode_n_matricization(X,j);
            Mn = Xn * Zn;
            % compute new core matrix
            cores{j} = Mn * pinv(Sn);
        end

        % normalize cores{j} and obtain lambda(j)
        for r = 1:R
            norm_r = 1;
            for j = 1:d
                nr = norm(cores{j}(:,r));
                cores{j}(:,r) = cores{j}(:,r) / nr;   % normalize 
                norm_r = norm_r * nr;
            end
            lambda(r) = norm_r;
        end
        
        % check for err
        err = 0;
        for j = 1:d
            err = err + norm(cores{j} - cores_old{j}, 'fro');
        end
        it = it + 1;
        fprintf('Iteration: %d, error: %d\n', it, err);
    end
end