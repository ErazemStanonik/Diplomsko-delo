function bestAcc = testHrSTM_main(trainX,trainY,testX,testY,num_classes)
% This is the main function for testing Higher rank STM model 
% over real life datasets
d = ndims(trainX) - 1;
num_test_samples = length(testY);
idx = repmat({':'},1,d);
tempTrainY = trainY;

cRange = [0.01 0.1 1 10 100];
rRange = [1 2 3];
bestAcc = 0;

% we check for the best accuracy
for C = cRange
    for R = rRange
        % we now train model for One VS Rest (OvR) classification
        % we build a casscade of HrSTMs
        HrSTMs = cell(1,num_classes);
        tic;
        for i = 1:num_classes
            % preprocess the labels for training 
            tempTrainY(trainY ~= i) = -1;
            tempTrainY(trainY == i) = 1;
        
            [W,b] = HrSTM(trainX,tempTrainY,'fitcsvm',R,C,1e-3,20);
            HrSTMs{i} = @(X) sign(innerprod(W,X) + b);
        end
        fprintf('Building a casscade of HrSTMs took ');
        toc;
        
        % now we test using accumulator vector, that counts number of votes in each
        % step in casscade of HrSTMs.
        correct = 0;
        for i = 1:num_test_samples
            A = zeros(num_classes,1);
            for j = 1:num_classes
                res = HrSTMs{j}(testX(idx{:},i));
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
        fprintf('*** HrSTM Accuracy for C = %.2f and R = %d is %.4f ***\n', C,R,accuracy);
    end
end