function [W] = cpToTensor(cores, lambda, R, shape)
    % This is a helper function that transforms CP decomposition of a
    % tensor back to its original tensor
    d = length(cores);
    W = zeros(shape);
    for r=1:R
        w = cell(1,d);
        for i =1:d
            w{i} = cores{i}(:,r);
        end
        W = W + lambda(r) * toTensor(w);
    end
end