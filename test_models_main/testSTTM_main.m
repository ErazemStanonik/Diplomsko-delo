function bestAcc = testSTTM_main(trainX,trainY,testX,testY,num_classes)
% This is the main function for testing Support tensor Train Machine model 
% over real life datasets
d = ndims(trainX) - 1;
num_test_samples = length(testY);
num_train_samples = length(trainY);
tempTrainY = trainY;
img_size = testX.size(1:d);

% we must preprocess the samples to TT format
% we stack all samples together, so they have the same first d TT cores
X = tensor(zeros([img_size, num_train_samples+num_test_samples]));
X(:,:,:,1:num_train_samples) = trainX;
X(:,:,:,num_train_samples+1:end) = testX;

%epsRange = [0.25, 0.125, 0.1, 0.0875, 0.075, 0.05];
epsRange = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001];

for eps = epsRange
    tic;
    ttX = tt_tensor(X.data,eps);
    fprintf("Samples converted to Tensor Train format in ");
    toc;
    % we extract some useful features that we will use later on
    n = ttX.n(1:end-1);                  % size of sample
    ranks = ttX.r(1:end-1);              % TT ranks
    ranks(end) = 1;
    ps = ttX.ps(1:end-1);                % positions of core tensors in ttX.core
    ps(end) = ps(end-1)+ranks(end-1)*n(end);
    % reshape ttX.core to d+1 3D core tensors.
    coresX = reshape_cores(ttX);
    % cores 1 ... d are equal for all samples. To obtain i-th samples we
    % must just extract i-th column from (d+1)-th core.
    samples = cell(1,num_train_samples+num_test_samples);
    shape = [ranks(d),n(d),ttX.r(d+1)];
    for i = 1:num_train_samples+num_test_samples
        samples{i} = coresX(1:d);
        samples{i}{d} = ttm(tensor(coresX{d},shape),coresX{d+1}(:,i,:)',3).data;
    end
    
    % if sample is lets say 32x32x3 image, then QR decompositino won't
    % work in last core (r_k x 3) so we must multiply the last core 
    % with the second to last
    for i = 1:num_train_samples+num_test_samples
        samples{i}{d-1} = ttm(tensor(samples{i}{d-1}), samples{i}{d}', 3).data;
        samples{i}{d} = eye(3);     % Identity matrix just to preserve tensor shape
    end
    ranks(end-1) = 3;
    disp(ranks);
    ps(end-1) = ps(end-2) + ranks(end-2)*n(end-1)*3;
    ps(end) = ps(end-1) + 3*3;
    
    trainX = samples(1:num_train_samples);
    testX = samples(num_train_samples+1:end);
    % for testing accuracy we must convert testX to tt_tensor format
    for i = 1:num_test_samples
        testX{i} = tt_core_to_tt_tensor(testX{i},n,ranks,d,ps);
    end
    
    bestAcc = 0;
    cRange = [0.01, 0.1, 1, 10, 100];            % from previous test i concluded, this is the best value.
    % NOW we can begin
    % we check for the best accuracy
    for C = cRange
        % we now train model for One VS Rest (OvR) classification
        % we build a casscade of STTMs
        STTMs = cell(1,num_classes);
        tic;
        for i = 1:num_classes
            % preprocess the labels for training 
            tempTrainY(trainY ~= i) = -1;
            tempTrainY(trainY == i) = 1;
            [W,b] = STTM(trainX,tempTrainY,n,ranks,d,1e-3,20);
            
            W = tt_core_to_tt_tensor(W,n,ranks,d,ps);
            STTMs{i} = @(X) sign(dot(W,X) + b);
        end
        fprintf('Building a casscade of STTMs took ');
        toc;
            
        % now we test using accumulator vector, that counts number of votes in each
        % step in casscade of STTMs.
        correct = 0;
        for i = 1:num_test_samples
            A = zeros(num_classes,1);
            for j = 1:num_classes
                res = STTMs{j}(testX{i});
                if res == 1
                    A(j) = A(j) + 1;
                end
            end
            [~,ix] = max(A);
            correct = correct + (ix == testY(i));
        end
        accuracy = correct / num_test_samples;
        if bestAcc < accuracy
            bestAcc = accuracy;
        end
        fprintf('*** STTM Accuracy for eps = %.3f C = %.2f is %.4f ***\n', eps,C,accuracy);
    end
end