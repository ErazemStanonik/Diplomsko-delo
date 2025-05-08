function accuracy = testR1LR_main(trainX,trainY,testX,testY,num_classes)
% This is the main function that tests rank-1 LR model over real life
% datasets

d = ndims(testX) - 1;
num_test_samples = length(testY);
idx = repmat({':'},1,d);
tempTrainY = trainY;

% we now train model for One VS Rest (OvR) classification
% we build a casscade of rank-1 LRs
r1LRs = cell(1,num_classes);
tic;
for i = 1:num_classes
    % preprocess the labels for training 
    tempTrainY(trainY ~= i) = -1;
    tempTrainY(trainY == i) = 1;

    [W,b] = r1LR(trainX, tempTrainY, 'fitclinear', 2.5e-1, 30);
    r1LRs{i} = @(X) 1 / (1 + exp(-(innerprod(W,X)+b)));
end
fprintf('Building a casscade of rank-1 LRs took ');
toc;

% now we test using accumulator vector, that counts number of votes in each
% step in casscade of rank-1 LRs.
correct = 0;
for i = 1:num_test_samples
    A = zeros(num_classes,1);
    for j = 1:num_classes
        res = r1LRs{j}(testX(idx{:},i));
        if sign(res-0.5) == 1
            A(j) = A(j) + 1;
        end
    end
    [~,ix] = max(A);
    correct = correct + (ix == testY(i));
end
accuracy = correct / num_test_samples;