classdef testCalculatePj < matlab.unittest.TestCase

    properties
        factors
    end

    methods(TestClassSetup)
        function addParentToPath(testCase)
            addpath('../utils');
        end
    end

    methods(TestMethodSetup)
        function initData(testCase)
            testCase.factors = cell(1,3);
            testCase.factors{1} = eye(2);
            testCase.factors{2} = [7 8; 9 10];
            testCase.factors{3} = [1 2; 3 4];
        end
    end

    methods (Test)
        function testCase1(testCase)
            actual = calculate_pj(testCase.factors,1,3);
            expected = [7 14 8 16; 21 28 24 32; 9 18 10 20; 27 36 30 40];
            testCase.verifyEqual(actual, expected);
        end

        function testCase2(testCase)
            actual = calculate_pj(testCase.factors,2,3);
            expected = [1 2 0 0; 3 4 0 0; 0 0 1 2; 0 0 3 4];
            testCase.verifyEqual(actual, expected);
        end

        function testCase3(testCase)
            actual = calculate_pj(testCase.factors,3,3);
            expected = [7 8 0 0; 9 10 0 0; 0 0 7 8; 0 0 9 10];
            testCase.verifyEqual(actual, expected);
        end
    end
end