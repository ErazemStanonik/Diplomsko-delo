function Mn = mode_n_matricization(X, n)
    % this is a helper function that makes a mode-n-matricization of a tensor X
    % along its mode n
    sz = size(X);
    order = [n, 1:n-1, n+1:length(sz)];
    Mn = reshape(permute(X, order), sz(n), []);
end