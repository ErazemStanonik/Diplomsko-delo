function [w, b, support_vectors] = mySVM(X, Y, kernel, C)
    % this function trains a SVM using convex optimization package cvx
    % X is of shape number_of_samples x vector_length, 
    % Y is of shape number_of_samples x 1
    % check if both have the same nuber of observations
    sizeX = size(X,1);
    sizeY = length(Y);

    assert(sizeX == sizeY, 'Error: Number of classes should be the same to number of observations.');

    n = size(X, 1);
    % TODO kernel K. For now linear kernel.
    K = (X * X');

    cvx_begin;
        cvx_quiet true;
        variables alphas(n);
        maximize( sum(alphas) - 0.5 * (alphas.*Y)'*K*(alphas.*Y) ); %sum(sum((alphas*alphas').*(Y*Y').*K)) )
        subject to;
            0 <= alphas <= C;
            sum(alphas.*Y) == 0;
    cvx_end;

    sv = find(alphas >= 1e-5,1);
    support_vectors = find(alphas > 1e-5);
    w = ((alphas.*Y)'*X)';
    b = Y(sv) - dot(w,X(sv,:));

end