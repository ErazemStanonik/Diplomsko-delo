function SVM = fitcSVM(X, Y)
    % this function trains a support vector machine
    % using fitcsvm function

    % check if both have the same nuber of observations
    sizeX = size(X,1);
    sizeY = length(Y);

    assert(sizeX == sizeY, 'Error: Number of classes should be the same to number of observations.');

    SVM = fitcsvm(X,Y);
end