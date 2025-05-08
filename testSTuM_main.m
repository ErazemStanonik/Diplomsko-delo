function bestAcc = testSTuM_main(trainX,trainY,testX,testY,num_classes)
% This is the main function that tests Support Tucker Machine over real
% life datasets.
d = ndims(trainX) - 1;
num_test_samples = length(testY);
idx = repmat({':'},1,d);
tempTrainY = trainY;

cRange = [0.01 0.1 1 10 100];
rRange = [1 2 3];
% now we find maximal rank Rk, which is defined as rank(X_(k)), that we can
% pass to STuM. It is acctually the least over all ranks
% Rk = zeros(d,1);
% for k = 1:d
%     minRk = Inf;
%     for i = 1:num_test_samples
%         rk = rank(mode_n_matricization(testX(idx{:},i), k));
%         if rk < minRk
%             minRk = rk;
%         end
%     end
%     Rk(k) = minRk;
% end

% check for the best accuracy
bestAcc = 0;
for C = cRange
    for R = rRange
        % we build a casscade of STuM classifiers
        STuMs = cell(1,num_classes);
        tic;
        for i = 1:num_classes
            % preprocess the labels for training 
            tempTrainY(trainY ~= i) = -1;
            tempTrainY(trainY == i) = 1;
        
            [W,b] = STuM(trainX,tempTrainY,'fitcsvm',R*ones(1,d),C,1e-3,20);
            STuMs{i} = @(X) sign(innerprod(W,X) + b);
        end

        fprintf('Building a casscade of STuMs took ');
        toc;
        
        % now we test using accumulator vector, that counts number of votes in each
        % step in casscade of STuMs.
        correct = 0;
        for i = 1:num_test_samples
            A = zeros(num_classes,1);
            for j = 1:num_classes
                res = STuMs{j}(testX(idx{:},i));
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
        fprintf('Accuracy for C = %.2f and R = %d is %.4f\n', C,R,accuracy);
    end
end