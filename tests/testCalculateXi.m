classdef testCalculateXi < matlab.unittest.TestCase

    properties
        X, w
    end

    methods(TestClassSetup)
        function addParentToPath(testCase)
            addpath('../utils');
        end
    end

    methods(TestMethodSetup)
        function initData(testCase)
            testCase.X = tensor(zeros(3,3,2));
            testCase.X(:,:,1) = [1 4 5; 2 8 7; 9 5 3];
            testCase.X(:,:,2) = [2 6 2; 8 1 3; 7 5 6];
            testCase.w = {[1 0 0]', [1 2 3]', [0 1]'};
        end
    end

    methods (Test)
        function testCase1(testCase)
            actual = calculate_xi(testCase.X,testCase.w,3,1);
            expected = [20 19 35]';
            testCase.verifyEqual(actual, expected);
        end

        function testCase2(testCase)
            actual = calculate_xi(testCase.X,testCase.w,3,2);
            expected = [2 6 2]';
            testCase.verifyEqual(actual, expected);
        end

        function testCase3(testCase)
            actual = calculate_xi(testCase.X,testCase.w,3,3);
            expected = [24 20]';
            testCase.verifyEqual(actual, expected);
        end
    end
end