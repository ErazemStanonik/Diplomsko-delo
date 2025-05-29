function bestAcc = testLTR_main(trainX,trainY,testX,testY,num_classes)
% This is the main function that tests Logistic Tensor Regression model on 
% real life datasets

d = ndims(trainX) - 1;
num_test_samples = length(testY);
idx = repmat({':'},1,d);
tempTrainY = trainY;

rRange = [1 2 3];
bestAcc = 0;

% we check for best accuracy
for R = rRange
    % we now train model for One VS Rest (OvR) classification
    % we build a casscade of LTRs
    LTRs = cell(1,num_classes);
    tic;
    for i = 1:num_classes
        % preprocess the labels for training 
        tempTrainY(trainY ~= i) = -1;
        tempTrainY(trainY == i) = 1;
        
        [W,b] = LTR(trainX,tempTrainY,R,1e-3,20);
        LTRs{i} = @(X) 1 / (1 + exp(-(innerprod(W,X)+b)));
    end
    fprintf('Building a casscade of LTRs took ');
    toc;

    % now we test using accumulator vector, that counts number of votes in each
    % step in casscade of LTRs.
    correct = 0;
    for i = 1:num_test_samples
        A = zeros(num_classes,1);
        for j = 1:num_classes
            res = LTRs{j}(testX(idx{:},i));
            if sign(res-0.5) == 1
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
    fprintf('*** LTR Accuracy for R = %.2f is %.4f ***\n', R,accuracy);
end