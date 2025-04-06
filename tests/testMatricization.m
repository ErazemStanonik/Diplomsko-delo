classdef testMatricization < matlab.unittest.TestCase
    
    properties
        X
    end

    methods(TestClassSetup)
        function addParentToPath(testCase)
            addpath('../utils');
        end
    end

    methods(TestMethodSetup)
        function initData(testCase)
            testCase.X(:,:,1) = [1 4 5; 2 8 7; 9 5 3];
            testCase.X(:,:,2) = [2 6 2; 8 1 3; 7 5 6];
        end
    end

    methods (Test)
        function testCase1(testCase)
            actual = mode_n_matricization(testCase.X,1);
            expected = [1 4 5 2 6 2; 2 8 7 8 1 3; 9 5 3 7 5 6];
            testCase.verifyEqual(actual, expected);
        end

        function testCase2(testCase)
            actual = mode_n_matricization(testCase.X,2);
            expected = [1 2 9 2 8 7; 4 8 5 6 1 5; 5 7 3 2 3 6];
            testCase.verifyEqual(actual, expected);
        end

        function testCase3(testCase)
            actual = mode_n_matricization(testCase.X,3);
            expected = [1 2 9 4 8 5 5 7 3; 2 8 7 6 1 5 2 3 6];
            testCase.verifyEqual(actual, expected);
        end
    end
end