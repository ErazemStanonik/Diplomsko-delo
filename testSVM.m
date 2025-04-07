function accuracy = testSVM(X,Y,k)
    % This method will test the average accuracy of the basic SVM method
    % using k-fold cross validation.
    
    sz = size(X);
    d = length(sz) - 1;
    num_samples = length(Y);

    group_size = num_samples / k;

    % we have to vectorize the data
    vecX = zeros(num_samples,prod(sz(1:d)));
    for m = 1:num_samples
        vecX(m,:) = vec(X(:,:,m))';
    end

    accuracy = 0;

    fprintf('Starting the %d-fold cross validation ...\n', k);
    % we now perform k-fold cross validation
    for i = 1:k
        test_idx = (i-1)*group_size+(1:group_size);
        train_idx = setdiff(1:num_samples,test_idx);
        % split the data
        trainX = vecX(train_idx,:);
        trainY = Y(train_idx);
        testX = vecX(test_idx,:);
        testY = Y(test_idx);

        SVM = fitcSVM(trainX,trainY);
        w = SVM.Beta;
        b = SVM.Bias;

        fun = @(x) sign(x*w + b);

        % we now check the accuracy
        t = 0;
        for j = 1:group_size
            res = fun(testX(j,:));
            if res == testY(j)
                t = t + 1;
            end
        end
        acc = t / group_size;
        fprintf('Accuracy in fold %d is %.2f.\n', i, acc);
        accuracy = accuracy + acc;
    end
    accuracy = accuracy / k;
    fprintf('Overall SVM ACCURACY = %.2f\n', accuracy);
end