% parts of this code were taken from 
% https://www.mathworks.com/help/stats/fitcsvm.html

load fisheriris;
inds = ~strcmp(species,'versicolor');
X = meas(inds,1:2);
s = species(inds);

SVMModel = fitcsvm(X,s);

sv = SVMModel.SupportVectors; % Support vectors
beta = SVMModel.Beta; % Linear predictor coefficients
b = SVMModel.Bias; % Bias term
X1 = linspace(min(X(:,1)),max(X(:,1)),100); % Separating hyperplane
X2 = -(beta(1)/beta(2)*X1)-b/beta(2);
m = 1/sqrt(beta(1)^2 + beta(2)^2);  % Margin half-width
X1margin_low = X1+beta(1)*m^2;
X2margin_low = X2+beta(2)*m^2;
X1margin_high = X1-beta(1)*m^2;
X2margin_high = X2-beta(2)*m^2;

% figure(1);
% clf;
% hold on
% gscatter(X(:,1),X(:,2),s);
% plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
% plot(X1,X2,'-');
% plot(X1margin_high,X2margin_high,'b--');
% plot(X1margin_low,X2margin_low,'r--');
% xlabel('X_1 (Sepal Length in cm)');
% ylabel('X_2 (Sepal Width in cm)');
% legend('setosa','virginica','Support Vector', ...
%     'Boundary Line','Upper Margin','Lower Margin');
% hold off

% now my function of SVM
Y = strcmp(s, 'setosa')*-2 + 1;  % Y is now either -1 or 1

[w,b,sv_index] = mySVM(X,Y,'linear',1);

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
legend('-1', '1', 'SV', 'Upper Margin', 'Lower Margin');
hold off;
