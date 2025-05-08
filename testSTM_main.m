function bestAcc = testSTM_main(trainX,trainY,testX,testY,num_classes)
% This is the main function for testing STM model over real databases

d = ndims(trainX) - 1;
num_test_samples = length(testY);
idx = repmat({':'},1,d);
tempTrainY = trainY;

cRange = [0.01 0.1 1 10 100];
bestAcc = 0;

% we check for the best accuracy
for C = cRange
    % we now train model for One VS Rest (OvR) classification
    % we build a casscade of STMs
    STMs = cell(1,num_classes);
    tic;
    for i = 1:num_classes
        % preprocess the labels for training 
        tempTrainY(trainY ~= i) = -1;
        tempTrainY(trainY == i) = 1;
    
        [W,b] = STM(trainX,tempTrainY,'fitcsvm',C,1e-3,20);
        STMs{i} = @(X) sign(innerprod(W,X) + b);
    end
    fprintf('Building a casscade of STMs took ');
    toc;
    
    % now we test using accumulator vector, that counts number of votes in each
    % step in casscade of STMs.
    correct = 0;
    for i = 1:num_test_samples
        A = zeros(num_classes,1);
        for j = 1:num_classes
            res = STMs{j}(testX(idx{:},i));
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
    fprintf('Accuracy for C = %.2f is %.4f\n', C, accuracy);
end