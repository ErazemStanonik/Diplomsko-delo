function X = toTensor(w)
% function toTensor performs outer product over d vectors in w. It returns
% a d-way tensor X.
% w is a cell that holds vectors.
    
    d = size(w,2);
    X = w{1};

    % we iterate through w and perform outer product
    % in each step we transform vector to 1 x 1 x...x length(w{j}) tensor
    % the j-th vector in w has j-1 1s before it.
    for j = 2:d
        X = X .* reshape(w{j}, [ones(1,j-1), length(w{j})]);
    end
end