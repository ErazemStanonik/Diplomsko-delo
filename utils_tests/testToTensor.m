classdef testToTensor < matlab.unittest.TestCase
    methods(TestClassSetup)
        function addParentToPath(testCase)
            addpath('../utils');
        end
    end

    methods (Test)
        function testCase1(testCase)
            w = {[1 2 3]', [4 5 6]'};
            actual = toTensor(w);
            expected = [4 5 6; 8 10 12; 12 15 18];
            testCase.verifyEqual(actual, expected);
        end

        function testCase2(testCase)
            w = {[1 3 5]', [2 4 6]'};
            actual = toTensor(w);
            expected = [2 4 6; 6 12 18; 10 20 30];
            testCase.verifyEqual(actual, expected);
        end

        function testCase3(testCase)
            w = {[1 2 3]', [4 5 6]', [10 20 30]'};
            actual = toTensor(w);
            expected(:,:,1) = [40 50 60; 80 100 120; 120 150 180];
            expected(:,:,2) = [80 100 120; 160 200 240; 240 300 360];
            expected(:,:,3) = [120 150 180; 240 300 360; 360 450 540];
            testCase.verifyEqual(actual, expected);
        end

        function testCase4(testCase)
            w = {[1 3 5]', [2 4 6]', [1 0 1]'};
            actual = toTensor(w);
            expected(:,:,1) = [2 4 6; 6 12 18; 10 20 30];
            expected(:,:,2) = zeros(3,3);
            expected(:,:,3) = [2 4 6; 6 12 18; 10 20 30];
            testCase.verifyEqual(actual, expected);
        end
    end
end