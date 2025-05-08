function [w,b] = myLR(vecX,Y)
% this function trains a LR model using convex optimization package cvx
% X is of shape number_of_samples x vector_length, 
% Y is of shape number_of_samples x 1

[m,feature_num] = size(vecX);

cvx_begin
    cvx_quiet true
    variables w(feature_num,1)
    variables b
    expression likelihood(m)
    for i = 1:m
        likelihood(i) = log(1 + exp(-Y(i) * (vecX(i,:)*w+b)));
    end
    minimize(sum(likelihood))
cvx_end