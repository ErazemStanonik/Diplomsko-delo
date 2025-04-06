function Pj = calculate_pj(factors, j, d)
    % This is a helper function that calculates Kronecker product of factor
    % matrices used Tucker decomposition with exception to j-th factor
    % matrix.
    
    Pj = 1;
    for k = setdiff(1:d,j)
        Pj = kron(Pj, factors{k});
    end
end

