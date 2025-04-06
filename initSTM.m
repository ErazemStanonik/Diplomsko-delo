X(:,:,1) = -ones(2,2);
X(:,:,2) = -0.75*ones(2,2);
X(:,:,3) = -0.5*ones(2,2);
X(:,:,4) = 0.4*ones(2,2);
X(:,:,5) = 0.8*ones(2,2);
X(:,:,6) = ones(2,2);

Y = [-1 -1 -1 1 1 1]';

[W, b] = STM(X,Y,1,1e-3,10);

f = @(X) sign(sum(dot(W,X)) + b);

X1 = [0.1 0.1; 0.1 0.1];
X2 = [-0.5 -0.5; -0.5 0.-5];
X3 = [1 0.1; 0.8 0.2];
X4 = [0.1 -1; 0.3 -0.2];

predict1 = f(X1);
predict2 = f(X2);
predict3 = f(X3);
predict4 = f(X4);

fprintf('Matrix %s belongs to class %d.\n', mat2str(X1), predict1)
fprintf('Matrix %s belongs to class %d.\n', mat2str(X2), predict2)
fprintf('Matrix %s belongs to class %d.\n', mat2str(X3), predict3)
fprintf('Matrix %s belongs to class %d.\n', mat2str(X4), predict4)