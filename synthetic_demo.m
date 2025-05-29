[trainX,trainY] = synthetic_init_RGB(100,32,'white',1);
[testX,testY] = synthetic_init_RGB(10,32,'white',1);

dRange = [1, 2, 3];
sigmaRange = [0.01, 0.1, 1, 10, 100];

accuracySVM = testSVM_main(trainX,trainY,testX,testY,3,'linear');
disp('----------');
for d = dRange
    accuracyKpSVM = testSVM_main(trainX,trainY,testX,testY,3,'polynomial', d);
    disp('----------');
end
for sigma = sigmaRange
    accuracyKgSVM = testSVM_main(trainX,trainY,testX,testY,3,'rbf', sigma);
    disp('----------');
end
accuracySTM = testSTM_main(trainX,trainY,testX,testY,3);
disp('----------');
accuracyHrSTM = testHrSTM_main(trainX,trainY,testX,testY,3);
disp('----------');
accuracySTuM = testSTuM_main(trainX,trainY,testX,testY,3);
disp('----------');
accuracySTTM = testSTTM_main(trainX,trainY,testX,testY,3);
disp('----------');
accuracyKSTTMalin = testKSTTM_main(trainX,trainY,testX,testY,3,'linear','a');
disp('----------');
accuracyKSTTMplin = testKSTTM_main(trainX,trainY,testX,testY,3,'linear','p');
disp('----------');
accuracyKSTTMapoly = testKSTTM_main(trainX,trainY,testX,testY,3,'polynomial','a');
disp('----------');
accuracyKSTTMppoly = testKSTTM_main(trainX,trainY,testX,testY,3,'polynomial','p');
disp('----------');
accuracyKSTTMagauss = testKSTTM_main(trainX,trainY,testX,testY,3,'gaussian','a');
disp('----------');
accuracyKSTTMpgaus = testKSTTM_main(trainX,trainY,testX,testY,3,'gaussian','p');
disp('----------');
accuracyLR = testLR_main(trainX,trainY,testX,testY,3);
disp('----------');
accuracyR1LR = testR1LR_main(trainX,trainY,testX,testY,3);
disp('----------');
accuracyLTR = testLTR_main(trainX,trainY,testX,testY,3);