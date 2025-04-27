function accuracies = testModels(X,Y,k,it)
    % This is the main testing function that tests all the models for given
    % samples X and corresponding labels Y. It does that with k-fold cross
    % validation.
    
    % we average results over 5 iterations
    accuracies = zeros(1,4);
    for i = 1:it
        accuracies(1) = accuracies(1) + testSVM(X,Y,k);
        accuracies(2) = accuracies(2) + testSTM(X,Y,k);
        accuracies(3) = accuracies(3) + testHrSTM(X,Y,k);
        accuracies(4) = accuracies(4) + testSTuM(X,Y,k);
    end
    accuracies = accuracies / it;
end