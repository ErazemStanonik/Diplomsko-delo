function cores = reshape_cores(ttX)
% This is a helper function that transforms ttX.core, which contains all
% cores in one big 1D array, to d 3D TT cores 

dims = ttX.n;
ranks = ttX.r;
d = length(dims);
pos = ttX.ps;
core = ttX.core;

cores = cell(1,d);

% we now reshape cores
for i = 1:d
    start = pos(i);
    stop = pos(i+1)-1;
    cores{i} = reshape(core(start:stop), [ranks(i),dims(i),ranks(i+1)]);
end