function x_i = calculate_xi(X, w, d, j)

    % this is a hleper function to calculate x_i
    % we calculate x_i as X mode-l-product with w(l), where l goes
    % from 1 to d without j, because we optimize for w_j
    
    for l = setdiff(1:d, j)
        sz = size(X);
        Xi = mode_n_matricization(X, l);
        Xi = w{l}' * Xi;       % this is mode-l-matricization. We have to transform it back

        if l == 1
            order = num2cell(sz(2:d));      % i couldn't find any other way to pass 'vector' to reshape function
            Xi = reshape(Xi, 1, order{:});
        elseif l == d
            order = num2cell(sz(1:d-1));
            Xi = reshape(Xi, order{:}, 1);
        else
            order1 = num2cell(sz(1:l-1));
            order2 = num2cell(sz(l+1:d));
            Xi = reshape(Xi, order1{:}, 1, order2{:});
        end
        X = tensor(Xi);
    end
    x_i = vec(Xi);
end