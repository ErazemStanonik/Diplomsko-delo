function [bestR] = find_best_R(trainX,trainY,testX,testY,num_classes)
% This function will find optimal R and C for the given data X.

maxAcc = 0;
bestR = 0;
R = zeros(1,10);

num_test_samples = length(testY);

for r = 1:10
    % we now train model for One VS Rest (OvR) classification
    % we build a casscade of HrSTMs
    HrSTMs = cell(1,num_classes);
    for i = 1:num_classes
        % preprocess the labels for training 
        tempTrainY(trainY ~= i) = -1;
        tempTrainY(trainY == i) = 1;
     
        [W,b] = HrSTM(trainX,tempTrainY,'fitcsvm',r,0.10,1e-3,50);
        HrSTMs{i} = @(X) sign(innerprod(W,X) + b);
    end
       
    % now we test using accumulator vector, that counts number of votes in each
    % step in casscade of HrSTMs.
    correct = 0;
    for i = 1:num_test_samples
        A = zeros(num_classes,1);
        for j = 1:num_classes
            res = HrSTMs{j}(testX(:,:,:,i));
            if res == 1
                A(j) = A(j) + 1;
            end
        end
        [~,ix] = max(A);
        correct = correct + (ix == testY(i));
    end
    accuracy = correct / num_test_samples;
    fprintf('Accuracy for R = %d is %.4f\n', r, accuracy);
    
    % we now save this accuracy
    R(r) = accuracy;
    if accuracy > maxAcc
        maxAcc = accuracy;
        bestR = r;
    end
end
hold on;
grid on;
plot(1:10, R, 'b*');
hold off;