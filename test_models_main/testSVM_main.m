function bestAcc = testSVM_main(trainX,trainY,testX,testY,num_classes,kernel,options)
% This is the main function for testing SVM model over real databases like
% KTH, egg_broken, ect.

d = ndims(trainX) - 1;
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

cRange = [0.01 0.1 1 10 100];
bestAcc = 0;

% we check for the best accuracy
for C = cRange
    % we now train model for One VS Rest (OvR) classification
    % we build a casscade of SVMs
    SVMs = cell(1,num_classes);
    tic;
    for i = 1:num_classes
        % preprocess the labels for training 
        tempTrainY(trainY ~= i) = -1;
        tempTrainY(trainY == i) = 1;
    
        if strcmp(kernel,'rbf')
            SVM = fitcsvm(vecTrainX,tempTrainY,'KernelFunction',kernel,'KernelScale',options,'BoxConstraint',C);
        elseif strcmp(kernel, 'polynomial')
            SVM = fitcsvm(vecTrainX,tempTrainY,'KernelFunction',kernel,'PolynomialOrder',options,'BoxConstraint',C);
        else
            SVM = fitcsvm(vecTrainX,tempTrainY,'KernelFunction','linear','BoxConstraint',C);
        end
        %SVMs{i} = @(X) sign(X*SVM.Beta + SVM.Bias);
        SVMs{i} = @(X) predict(SVM,X);
    end
    fprintf('Building a casscade of SVMs took ');
    toc;
    
    % now we test using accumulator vector, that counts number of votes in each
    % step in casscade of SVMs.
    correct = 0;
    for i = 1:num_test_samples
        A = zeros(num_classes,1);
        for j = 1:num_classes
            res = SVMs{j}(vecTestX(i,:));
            if res == 1
                A(j) = A(j) + 1;
            end
        end
        [~,idx] = max(A);
        correct = correct + (idx == testY(i));
    end
    accuracy = correct / num_test_samples;
    if bestAcc < accuracy
        bestAcc = accuracy;
    end
    fprintf('*** Accuracy for C = %.2f is %.4f ***\n', C, accuracy);
end