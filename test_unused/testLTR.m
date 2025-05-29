function accuracy = testLTR(X,Y,k)
% This function tests Logistic Tensor Regression using k-fold cross
% validation

d = ndims(X) - 1;
num_samples = length(Y);

group_size = num_samples / k;
idx = repmat({':'},1,d);

accuracy = 0;
fprintf('Starting the %d-fold cross validation ...\n', k);
for i = 1:k
    test_idx = (i-1)*group_size+(1:group_size);
    train_idx = setdiff(1:num_samples,test_idx);
    % split the data
    trainX = X(idx{:},train_idx);
    trainY = Y(train_idx);
    testX = X(idx{:},test_idx);
    testY = Y(test_idx);

    [W,b] = LTR(trainX,trainY,1,1e-3,20);
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
fprintf('Overall LTR ACCURACY = %.2f\n', accuracy);