function bestAcc = testKSTTM_main(trainX,trainY,testX,testY,num_classes,kernel, flag)
% This is the main function that will test kernelized STTM model over real 
% life datasets

% data preparation
d = ndims(trainX) - 1;
num_test_samples = length(testY);
num_train_samples = length(trainY);
tempTrainY = trainY;
img_size = testX.size(1:d);

% preprocess data to a TT format
ttX = tensor(zeros([img_size, num_train_samples+num_test_samples]));
ttX(:,:,:,1:num_train_samples) = trainX;
ttX(:,:,:,num_train_samples+1:end) = testX;
tic;
eps = 0.045;
ttX = tt_tensor(ttX.data,eps);
fprintf("Samples converted to Tensor Train format in ");
toc;
ttX.r
% some adjustments to the TT format
shape = [ttX.r(d),ttX.n(d),ttX.r(d+1)];
cores = reshape_cores(ttX);
cores{d} = ttm(tensor(cores{d},shape), cores{d+1}', 3).data;
cores(d+1) = [];                % cores is now a 1 x d cell with d-th TT core containing 'num_samples' frontal slices

trainX = cores;
testX = cores;
trainX{d} = cores{d}(:,:,1:num_train_samples);
testX{d} = cores{d}(:,:,num_train_samples+1: end);

cRange = [0.1, 1, 10, 100];
sigmaRange = [0.1, 1, 10, 100];
bestAcc = 0;

% now we begin with checking for the best accuracy
for sigma = sigmaRange
    % we can already build a kernel matrix, which only depends on a
    % parameter sigma
    tic;
    K = kernel_mat(trainX, length(trainY), d, sigma, 1, flag,kernel);
    fprintf('Kernel matrix computed in ');
    toc;
    % to prevent if some eigenvalues are negative we set them to 0
    [V,D] = eig(K);
    if min(diag(D)) < 0
        disp('Some negaitve eigenvalues!');
        th = 1e-8;
        D(D < th) = 0;
        newK = V * D * V';
        fprintf('Reconstruction error: %d\n', norm(K - newK));
        K = newK;
    end
    % while debugging it turned out that some values might be complex, with
    % the imaginary component really smal - of order 1e-20
    K = real(K);
    for C = cRange
        % we now train model for One VS Rest (OvR) classification
        % we build a casscade of KSTTMs
        KSTTMs = cell(1,num_classes);
        tic;
        for i = 1:num_classes
             % preprocess the labels for training 
            tempTrainY(trainY ~= i) = -1;
            tempTrainY(trainY == i) = 1;
            
            [alpha, b] = svm_solver(K,tempTrainY,C,length(trainY));
            KSTTMs{i} = @(testX) predict(testX, alpha, b, trainX, tempTrainY, sigma, 3, 1, flag,kernel);
        end
        fprintf('Finished building a casscade of classifiers in ');
        toc;

        % now we test 
        % becuse sign(predict) gives us +- 1 we multiply with class number,
        % and then we check how many are correct.
        correct = 0;
        for j = 1:num_classes
            correct = correct + sum(j * sign(KSTTMs{j}(testX)) == testY);
        end
        accuracy = correct / num_test_samples;
        if bestAcc < accuracy
            bestAcc = accuracy;
        end
        fprintf('Accuracy for C = %.2f and sigma = %.2f is %.4f\n', C,sigma,accuracy);
    end
end