function ttW = tt_core_to_tt_tensor(cores,n,ranks,d,ps)
% This is a helper function that transforms a cell of d TT core tensors
% back into a tt_tensor class. 
%
% Parameter cores is a cell containing d TT cores.
%
% parameters n, ranks, d, ps are used in tt_tensor class
ttW = tt_tensor();
ttW.d = d;
ttW.r = ranks;
ttW.n = n;
ttW.ps = ps;

core = [];
for i = 1:d
    core = [core; cores{i}(:)];
end
ttW.core = core;