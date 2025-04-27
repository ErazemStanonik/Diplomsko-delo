function [bestR] = find_best_R(X,Y,k)
% This function will find optimal R and C for the given data X.

maxAcc = 0;
bestR = 0;
R = zeros(1,10);
for r = 1:10
    acc = testHrSTM(X,Y,k,r,1.5);
    R(r) = acc;
    if acc > maxAcc
            maxAcc = acc;
            bestR = r;
    end
end
hold on;
grid on;
plot(1:10, R, 'b*');
hold off;