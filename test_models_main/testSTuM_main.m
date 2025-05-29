function bestAcc = testSTuM_main(trainX,trainY,testX,testY,num_classes)
% This is the main function that tests Support Tucker Machine over real
% life datasets.
d = ndims(trainX) - 1;
num_test_samples = length(testY);
idx = repmat({':'},1,d);
tempTrainY = trainY;

cRange = [0.01 0.1 1 10 100];
R1 = [1 2 3];
R2 = [1 2 3];
R3 = [1 2 3];
% now we find maximal rank Rk, which is defined as rank(X_(k)), that we can
% pass to STuM. It is acctually the least over all ranks
Rk = zeros(d,1);
for k = 1:d
    minRk = Inf;
    for i = 1:num_test_samples
        rk = rank(mode_n_matricization(testX(idx{:},i), k));
        if rk < minRk
            minRk = rk;
        end
    end
    Rk(k) = minRk;
end
% now we must check, that rRangeK doesn't exceed the minRk
% R1 = R1(R1 <= Rk(1));   r1 = length(R1);
% R2 = R2(R2 <= Rk(2));   r2 = length(R2);
% R3 = R3(R3 <= Rk(3));   r3 = length(R3);
% rRange = max(r1,max(r2,r3));
rRange = 3;
% check for the best accuracy
bestAcc = 0;
for C = cRange
    for r = 1:rRange
        % we build a casscade of STuM classifiers
        STuMs = cell(1,num_classes);
        tic;
        for i = 1:num_classes
            % preprocess the labels for training 
            tempTrainY(trainY ~= i) = -1;
            tempTrainY(trainY == i) = 1;
        
            % we use min() in indexing so that if for example R1 is
            % [1,2,3], R2 = [1,2,3] and R3 is only [1], we don't exceede
            % the number of elements. So we would have [1,1,1], [2,2,1], [3,3,1]. 
%             [W,b] = STuM(trainX,tempTrainY,'fitcsvm',[R1(min(r,r1)),R2(min(r,r2)),R3(min(r,r3))],C,1e-3,20);
            [W,b] = STuM(trainX,tempTrainY,'fitcsvm',[r,r,r],C,1e-3,20);
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
        fprintf('*** STuM Accuracy for C = %.2f and ranks = [%d, %d, %d] is %.4f ***\n', ...
            C,r,r,r,accuracy);
            %C,R1(min(r,r1)),R2(min(r,r2)),R3(min(r,r3)),accuracy);
    end
end