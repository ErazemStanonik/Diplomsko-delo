classdef testKhatriRao < matlab.unittest.TestCase
    properties
        cores
    end

    methods(TestClassSetup)
        function addParentToPath(testCase)
            addpath('../utils');
        end
    end

    methods(TestMethodSetup)
        function initData(testCase)
            testCase.cores = {[1 2 3; 4 5 6], [1 0 0; 0 1 0], [1 2 1], [2 4 5]};
        end
    end

    methods (Test)
        function testCase1(testCase)
            actual = khatriRao(testCase.cores,1,4,3);
            expected = [2 0 0; 0 8 0];
            testCase.verifyEqual(actual, expected);
        end

        function testCase2(testCase)
            actual = khatriRao(testCase.cores,2,4,3);
            expected = [2 16 15; 8 40 30];
            testCase.verifyEqual(actual, expected);
        end

        function testCase3(testCase)
            actual = khatriRao(testCase.cores,3,4,3);
            expected = [2 0 0; 8 0 0; 0 8 0; 0 20 0];
            testCase.verifyEqual(actual, expected);
        end

        function testCase4(testCase)
            actual = khatriRao(testCase.cores,4,4,3);
            expected = [1 0 0; 4 0 0; 0 4 0; 0 10 0];
            testCase.verifyEqual(actual, expected);
        end
    end
end