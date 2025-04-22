X = [0.8147    0.2785; 
     0.9058    0.5469;
     0.6324    0.1576;
     1.0000   -0.1325;
     0.9120   -0.6789;
     0.3001   -1.0201;
     0.9134    0.9649;

     0.1270    0.9575;
    -0.4500    0.6660; 
     0.0975    0.9706;
     0.0002    1.3943; 
    -0.4562    1.6666;
     0.7500    1.9870 ];
Y = [-1 -1 -1 -1 -1 -1 -1 1 1 1 1 1 1]';

% firstly we call fitcsvm method on given data
SVM = fitcSVM(X, Y);

sv = SVM.SupportVectors;
w = SVM.Beta;
b = SVM.Bias;

% the best separating hyperplane is w'*x + b = 0
X1 = linspace(min(X(:,1)), max(X(:,1)));
X2 = (-w(1) * X1-b) / w(2);

m = 1/sqrt(w(1)^2 + w(2)^2);  % Margin half-width
X1margin_low = X1+w(1)*m^2;
X2margin_low = X2+w(2)*m^2;
X1margin_high = X1-w(1)*m^2;
X2margin_high = X2-w(2)*m^2;

figure(1);
clf;
hold on;
gscatter(X(:,1), X(:,2), Y);
plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 10);
plot(X1, X2, '-');
plot(X1margin_high,X2margin_high,'b--');
plot(X1margin_low,X2margin_low,'r--');
legend('-1', '1', 'Support Vector', 'Upper Margin', 'Lower Margin');
hold off;

% now we call mySVM on given data
[w, b, sv_index] = mySVM(X,Y,'linear', 100);

sv = X(sv_index,:);
X2 = (-w(1) * X1-b) / w(2);     % X1 stays the same

m = 1/sqrt(w(1)^2 + w(2)^2);  % Margin half-width
X1margin_low = X1+w(1)*m^2;
X2margin_low = X2+w(2)*m^2;
X1margin_high = X1-w(1)*m^2;
X2margin_high = X2-w(2)*m^2;

figure(2);
clf;
hold on;
gscatter(X(:,1), X(:,2), Y);
plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 10);
plot(X1, X2, '-');
plot(X1margin_high,X2margin_high,'b--');
plot(X1margin_low,X2margin_low,'r--');
legend('-1', '1', 'Upper Margin', 'Lower Margin');
hold off;