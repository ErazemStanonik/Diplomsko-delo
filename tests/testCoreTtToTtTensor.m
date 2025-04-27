classdef testCoreTtToTtTensor < matlab.unittest.TestCase

    properties
        ttX, cores
    end

    methods(TestClassSetup)
        function addParentToPath(testCase)
            addpath('../utils');
        end
    end

    methods(TestMethodSetup)
        function initData(testCase)
            testCase.ttX = tt_tensor();
            testCase.ttX.d = 4;
            testCase.ttX.ps = [1 7 19 27 31]';
            testCase.ttX.n = [3 3 2 2]';
            testCase.ttX.r = [1 2 2 2 1]';
            testCase.cores = cell(1,4);
            testCase.cores{1} = reshape([1 1 1 1 1 1], [1 3 2]);
            testCase.cores{2} = reshape([1 4 2 5 3 6 2 1 4 3 6 5], [2 3 2]); 
            testCase.cores{3} = reshape([1 0 0 1 0 1 1 0], [2 2 2]);
            testCase.cores{4} = reshape([2 6 4 8], [2 2 1]);    
        end
    end

    methods (Test)
        function testCase1(testCase)
            actual = tt_core_to_tt_tensor(testCase.ttX,testCase.cores);
            expected = [28 28 28 56 56 56 84 84 84 36 36 36 56 56 56 76 ...
                76 76 44 44 44 84 84 84 124 124 124 52 52 52 84 84 84 ... 
                116 116 116]';
            testCase.verifyEqual(full(actual), expected);
        end
    end
end