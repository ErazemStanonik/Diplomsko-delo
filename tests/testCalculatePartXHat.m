classdef testCalculatePartXHat < matlab.unittest.TestCase

    properties
        tt_cores, n, ranks, d
    end

    methods(TestClassSetup)
        function addParentToPath(testCase)
            addpath('../utils');
        end
    end

    methods(TestMethodSetup)
        function initData(testCase)
            testCase.tt_cores = cell(1,4);
            testCase.tt_cores{1} = reshape([1 1 1 1 1 1], [1 3 2]);
            testCase.tt_cores{2} = reshape([1 4 2 5 3 6 2 1 4 3 6 5], [2 3 2]);
            testCase.tt_cores{3} = reshape([1 0 0 1 0 1 1 0], [2 2 2]);
            testCase.tt_cores{4} = reshape([2 6 4 8], [2 2 1]);
            testCase.n = [3 3 2 2];
            testCase.ranks = [1 2 2 2 1];
        end
    end

    methods (Test)
        function testCase1(testCase)
            actual = calculate_part_x_hat(testCase.tt_cores,testCase.tt_cores,1,1,'L', ...
                testCase.n,testCase.ranks);
            expected = [3 3; 3 3];
            testCase.verifyEqual(actual, expected);
        end

        function testCase2(testCase)
            actual = calculate_part_x_hat(testCase.tt_cores,testCase.tt_cores,3,4,'R', ...
                testCase.n,testCase.ranks);
            expected = [120 88; 88 120];
            testCase.verifyEqual(actual, expected);
        end

        function testCase3(testCase)
            actual = calculate_part_x_hat(testCase.tt_cores,testCase.tt_cores,1,3,'L', ...
                testCase.n,testCase.ranks);
            expected = [1002 978; 978 1002];
            testCase.verifyEqual(actual, expected);
        end

        function testCase4(testCase)
            actual = calculate_part_x_hat(testCase.tt_cores,testCase.tt_cores,1,4,'L', ...
                testCase.n,testCase.ranks);
            expected = 206304;
            testCase.verifyEqual(actual, expected);
        end

        function testCase5(testCase)
            actual = calculate_part_x_hat(testCase.tt_cores,testCase.tt_cores,2,4,'R', ...
                testCase.n,testCase.ranks);
            expected = [13328 16688; 16688 22064];
            testCase.verifyEqual(actual, expected);
        end
    end
end