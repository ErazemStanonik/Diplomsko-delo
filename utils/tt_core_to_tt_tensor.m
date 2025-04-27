function ttW = tt_core_to_tt_tensor(ttX, cores)
% This is a helper function that transforms a cell of d TT core tensors
% back into a tt_tensor class. 
%
% Parameter ttX is the has the same properties (shape, size, ranks, ...) as
% our ttW, where we only change ttX.core parameter.
%
% Parameter cores is a cell containing d TT cores.

d = length(cores);
ttW = tt_tensor(ttX);
core = [];
for i = 1:d
    core = [core; cores{i}(:)];
end
ttW.core = core;