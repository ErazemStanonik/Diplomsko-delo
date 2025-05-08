function accuracy = testKSTTM(X,Y,k,kernel,flag)
% This function will test kernelised STTM.

% data preparation
d = ndims(X) - 1;
num_samples = length(Y);
img_size = X.size(1);
group_size = num_samples / k;

% X=permute(X,[3 1 2 4]).data;     % X is now RGB x img_size x img_size x num_samples
% 
% [u,s,v] = VBMF(reshape(X,size(X,1),[]), cacb, sigma2);
% TT{1}=reshape(u,[1 size(u,1) size(u,2)]);
% for t=2:d-1
%     [u,s,v] = VBMF(reshape(s*v',size(u,2)*size(X,t),[]), cacb, sigma2);
%     TT{t}=reshape(u,[size(u,1)/size(X,t) size(X,t) size(u,2)]);
% end
% TT{d}=reshape(s*v',size(u,2),size(X,3),size(X,4));
% U_common=reshape(TT{1},size(TT{1},1)*size(TT{1},2),size(TT{1},3))*reshape(TT{2},size(TT{2},1),size(TT{2},2)*size(TT{2},3));
% U_common=reshape(U_common,size(TT{1},2)*size(TT{2},2),size(TT{2},3));
% traindata_temp = reshape(X,[3*img_size,num_samples*img_size]);
% TT{d}=reshape(pinv(U_common)*traindata_temp,[size(TT{2},3),img_size,num_samples]);
% X = TT;

X = X.data;
ttX = tt_tensor(X, 1e-3);
cores = reshape_cores(ttX);
cores{d} = ttm(tensor(cores{d}), cores{d+1}', 3).data;
cores(d+1) = [];                % cores is now a 1 x d cell with d-th TT core containing 'num_samples' frontal slices
X = cores;

[C,sigma] = find_best_parameters_KSTTM(X,Y,kernel,flag);

fprintf('Starting the %d-fold cross validation ...\n', k);
accuracy = 0;
for i = 1:k
    test_idx = (i-1)*group_size+(1:group_size);
    train_idx = setdiff(1:num_samples,test_idx);
    % split samples
    trainX = X;
    testX = X;
    trainX{d} = X{d}(:,:,train_idx);
    testX{d} = X{d}(:,:,test_idx);
    trainY = Y(train_idx);
    testY = Y(test_idx);
    
    K = kernel_mat(trainX, length(trainY), d, sigma, 1, flag,kernel);
    [alpha, b] = svm_solver(K,trainY,C,length(trainY));
        
    prediction = predict(testX, alpha, b, trainX, trainY, sigma, 3, 1, flag,kernel);
    correct = sum(sign(prediction) == testY);

    acc = correct / group_size;

    fprintf('Accuracy in fold %d is %.2f.\n', 1, acc);
    accuracy = accuracy + acc;
end
accuracy = accuracy / k;
fprintf('Overall KSTTM ACCURACY = %.2f\n', accuracy);
end