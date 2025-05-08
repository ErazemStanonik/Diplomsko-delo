function accuracy = testR1LR(X,Y,solver,k)
% This function will test Rank-1 Logistic Regression using k-fold cross
% validation.

d = ndims(X) - 1;
num_samples = length(Y);

group_size = num_samples / k;

accuracy = 0;

fprintf('Starting the %d-fold cross validation ...\n', k);
% we now perform k-fold cross validation
for i = 1:k
    test_idx = (i-1)*group_size+(1:group_size);
    train_idx = setdiff(1:num_samples,test_idx);
    idx = repmat({':'},1,d);
    % split the data
    trainX = X(idx{:},train_idx);
    trainY = Y(train_idx);
    testX = X(idx{:},test_idx);
    testY = Y(test_idx);

    [W,b] = r1LR(trainX,trainY,solver,1e-3,30);
    fun = @(X) 1 / (1 + exp(-(innerprod(W,X)+b)));

    % we now check the accuracy
    t = 0;
    for j = 1:group_size
        res = fun(testX(idx{:},j));
        if sign(res-.5) == testY(j)
            t = t + 1;
        end
    end
    acc = t / group_size;
    fprintf('Accuracy in fold %d is %.2f.\n', i, acc);
    accuracy = accuracy + acc;
end
accuracy = accuracy / k;
fprintf('Overall LR ACCURACY = %.2f\n', accuracy);
end