classdef testCPtoTensor < matlab.unittest.TestCase
    properties
        cores, lambda, R, shape
    end

    methods(TestClassSetup)
        function addParentToPath(testCase)
            addpath('../utils');
        end
    end

    methods(TestMethodSetup)
        function initData(testCase)
            testCase.cores = {[1 2; 3 4; 5 6], [1 4; 2 5; 3 6], [1 0; 0 1]};
            testCase.lambda = [2 1];
            testCase.R = 2;
            testCase.shape = [3 3 2];
        end
    end

    methods (Test)
        function testCase1(testCase)
            actual = cpToTensor(testCase.cores,testCase.lambda, testCase.R, testCase.shape);
            expected(:,:,1) = [2 4 6; 6 12 18; 10 20 30];
            expected(:,:,2) = [8 10 12; 16 20 24; 24 30 36];
            testCase.verifyEqual(actual, expected);
        end
    end
end