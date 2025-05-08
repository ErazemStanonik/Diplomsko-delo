function [C, sigma] = find_best_parameters_KSTTM(X,Y,kernel,flag)
% This function finds best parameters C and sigma for k-STTM

d = length(X);
num_samples = length(Y);


% devide samples
train_size = round(0.7 * num_samples);
idx = randperm(num_samples);
train_idx = idx(1:train_size);
test_idx = idx(train_size+1:end);

trainX = X;
testX = X;
trainX{d} = X{d}(:,:,train_idx);
testX{d} = X{d}(:,:,test_idx);
trainY = Y(train_idx);
testY = Y(test_idx);

sigmaRange = [0.1, 1, 10, 100];
cRange = [1e-3, 1e-2, 1e-1, 1 10 100];

maxCorrect = 0;
bestC = 0;
bestSigma = 0;

for sigma = sigmaRange
    K = kernel_mat(trainX, length(trainY), d, sigma, 1, flag, kernel);
    for C = cRange
        [alpha, b] = svm_solver(K,trainY,C,length(trainY));
       
        prediction = predict(testX, alpha, b, trainX, trainY, sigma, 3, 1, flag, kernel);
        correct = sum(sign(prediction) == testY);
        if maxCorrect < correct
            bestC = C;
            bestSigma = sigma;
            maxCorrect = correct;
        end
    end
end
C = bestC;
sigma = bestSigma;
fprintf('Best parameters at accuracy %.3f\n', maxCorrect/(num_samples-train_size));