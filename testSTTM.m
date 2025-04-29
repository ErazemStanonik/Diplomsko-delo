function accuracy = testSTTM(X,Y,mode,solver,k,C)
    % This method will test the average accuracy of the STM method
    % using k-fold cross validation.  
    %
    % mode parameter tells us if samples are RGB pictuers
    
    num_samples = length(Y);

    group_size = num_samples / k;
    
    accuracy = 0;

    % Firstly we have to transform X to tt_tensor
    % X represents all d-way tensor samples stacked to a new (d+1)-way
    % tensor X. we compute TT decomposition over X so all samples have the
    % same first d TT cores.
    ttX = tt_tensor(X.data,1e-4);
    % we extract some useful features that we will use later on
    n = ttX.n(1:end-1);                  % size of sample
    ranks = ttX.r(1:end-1);              % TT ranks
    ranks(end) = 1;
    d = ttX.d - 1;                       % dimensions of sample in X
    ps = ttX.ps(1:end-1);                % positions of core tensors in ttX.core
    ps(end) = ps(end-1)+ranks(end-1)*n(end);
    % reshape ttX.core to d+1 3D core tensors.
    coresX = reshape_cores(ttX);
    % cores 1 ... d are equal for all samples. To obtain i-th samples we
    % must just extract i-th column from (d+1)-th core.
    samples = cell(1,num_samples);
    for i = 1:num_samples
        samples{i} = coresX(1:d);
        samples{i}{d} = ttm(tensor(coresX{d}),coresX{d+1}(:,i,:)',3).data;
    end

    % if sample is lets say 32x32x3 image, then QR decompositino won't
    % work in last core (r_k x 3) so we must multiply the last core 
    % with the second to last
    if strcmp(mode,'RGB')
        for i = 1:num_samples
            samples{i}{d-1} = ttm(tensor(samples{i}{d-1}), samples{i}{d}', 3).data;
            samples{i}{d} = eye(3);     % Identity matrix just to preserve tensor shape
        end
        ranks(end-1) = 3;
        ps(end-1) = ps(end-2) + ranks(end-2)*n(end-1)*3;
        ps(end) = ps(end-1) + 3*3;
    end

    fprintf('Starting the %d-fold cross validation ...\n', k);
    % we now perform k-fold cross validation
    for i = 1:k
        test_idx = (i-1)*group_size+(1:group_size);
        train_idx = setdiff(1:num_samples,test_idx);
        % split the data
        testX = samples(test_idx);
        testY = Y(test_idx);

        [W,b] = STTM(samples(train_idx),Y(train_idx),n,ranks,d,100,1e-3);

        % We have to transform cores in W to tt_tensor class ...
        ttW = tt_core_to_tt_tensor(W,n,ranks,d,ps);
        % ... and also testX
        for j = 1:group_size
            testX{j} = tt_core_to_tt_tensor(testX{j},n,ranks,d,ps);
        end        

        fun = @(X) sign(dot(ttW,X) + b);  % dot is a function over 2 Tensor trains

        % we now check the accuracy
        t = 0;
        for j = 1:group_size
            res = fun(testX{j});
            if res == testY(j)
                t = t + 1;
            end
        end
        acc = t / group_size;
        fprintf('Accuracy in fold %d is %.2f.\n', i, acc);
        accuracy = accuracy + acc;
    end
    accuracy = accuracy / k;
    fprintf('Overall HrSTM ACCURACY = %.2f\n', accuracy);
end