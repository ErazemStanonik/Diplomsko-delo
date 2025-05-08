function accuracy = testLR_main(trainX,trainY,testX,testY,num_classes)
% This is the main function for testing LR model over real databases like
% KTH, egg_broken, ect.

d = ndims(testX) - 1;
num_train_samples = length(trainY);
num_test_samples = length(testY);
img_size = trainX.size(1:end-1);
vecTrainX = zeros(num_train_samples, prod(img_size));
vecTestX = zeros(num_test_samples, prod(img_size));
idx = repmat({':'},1,d);

% vectorize the data
for i = 1:num_train_samples
    vecTrainX(i,:) = vec(trainX(idx{:},i))';
end
for i = 1:num_test_samples
    vecTestX(i,:) = vec(testX(idx{:},i))';
end
tempTrainY = trainY;
% we now train model for One VS Rest (OvR) classification
% we build a casscade of LRs
LRs = cell(1,num_classes);
tic;
for i = 1:num_classes
    % preprocess the labels for training 
    tempTrainY(trainY ~= i) = -1;
    tempTrainY(trainY == i) = 1;

    LR = fitclinear(vecTrainX, tempTrainY, 'Learner','logistic','Solver','lbfgs');
    LRs{i} = @(x) predict(LR, x);
end
fprintf('Building a casscade of LRs took ');
toc;

% now we test using accumulator vector, that counts number of votes in each
% step in casscade of LRs.
correct = 0;
for i = 1:num_test_samples
    A = zeros(num_classes,1);
    for j = 1:num_classes
        res = LRs{j}(vecTestX(i,:));
        if res == 1
            A(j) = A(j) + 1;
        end
    end
    [~,idx] = max(A);
    correct = correct + (idx == testY(i));
end
accuracy = correct / num_test_samples;