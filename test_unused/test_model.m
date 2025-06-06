function accuracy = test_model(X,Y,num_samples,ratio, model)
    % This is a function that trains and tests variations of SVM models.
    %
    % X is input tensor contaning num_samples samples
    %
    % Y is vector of labels
    %
    % ratio devides X to training and testing set.
    % 
    % model can be 'SVM', 'STM', 'HoSTM' or 'STuM'
    
    % we form indices that will devide X to training and test set ...
    train_size = round(ratio * num_samples);
    test_size = num_samples - train_size;
    idx = randperm(num_samples);
    train_idx = idx(1:train_size);
    test_idx = idx(train_size+1:end);

    % ... and we devide it.
    trainX = X(:,:,train_idx);
    trainY = Y(train_idx);
    testX = X(:,:,test_idx);
    testY = Y(test_idx);

    sz = size(X);
    d = length(sz) - 1;
    shape = sz(1:d);

    switch model
        case 'SVM'
            % we have to vectorize the samples X
            vecXtrain = zeros(train_size,prod(shape));
            vecXtest = zeros(test_size,prod(shape));
            for i = 1:train_size
                vecXtrain(i,:) = vec(trainX(:,:,i))';
            end
            for i = 1:test_size
                vecXtest(i,:) = vec(testX(:,:,i))';
            end
            
            % train our model
            SVM = fitcSVM(vecXtrain,trainY);
            w = SVM.Beta;
            b = SVM.Bias;

            fun = @(x) sign(x*w + b);
            % now we check how many correct classifications do we get
            t = 0;
            for j = 1:test_size
                res = fun(vecXtest(j,:));
                if res == testY(j)
                    t = t + 1;
                end
            end
            % we return model's accuracy
            accuracy = t/test_size;

        case 'STM'
            % no need to mess with input just train our STM
            [W,b] = STM(trainX,trainY,1,1e-5,10);
            W = tensor(W);

            fun = @(X) sign(innerprod(W,tensor(X)) + b);
            % now we check how many correct classifications do we get
            t = 0;
            for j = 1:test_size
                res = fun(testX(:,:,j));
                if res == testY(j)
                    t = t + 1;
                end
            end
            % we return model's accuracy
            accuracy = t/test_size;

        case 'HoSTM'
            % train our Higher Order Support Tensor Machine
            [W,b] = HoSTM(trainX,trainY,2,1,1e-5,10);
            W = tensor(W);

            % we must check if we can approximate W with rank R
            if b == -1
                accuracy = -1;
                return;
            end

            fun = @(X) sign(innerprod(W,tensor(X)) + b);
            % now we check how many correct classifications do we get
            t = 0;
            for j = 1:test_size
                res = fun(testX(:,:,j));
                if res == testY(j)
                    t = t + 1;
                end
            end
            % we return model's accuracy
            accuracy = t/test_size;

        case 'STuM'
            % train our Support Tucker Machine
            [W,b] = STuM(tensor(trainX),trainY,[2,2],1,10,1e-5);

            fun = @(X) sign(innerprod(W,tensor(X)) + b);
            % now we check how many correct classifications do we get
            t = 0;
            for j = 1:test_size
                res = fun(testX(:,:,j));
                if res == testY(j)
                    t = t + 1;
                end
            end
            % we return model's accuracy
            accuracy = t/test_size;

        otherwise
            fprintf('Incorrect model: %s\n', model);
            accuracy = -1;
    end

end