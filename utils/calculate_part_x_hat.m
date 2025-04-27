function X = calculate_part_x_hat(tt_cores, ttX, start, stop, part, n, ranks)
% This is a helper function that computes left or right side of inner
% product used in STTM computation of x_hat. See STTM
%
% tt_cores is a cell containing TT cores of W
% 
% ttX is a sample in TT format
%
% start and stop are indeces
%
% part is either 'L' or 'R'
%
% n = ttX.n;
% ranks = ttX.r;

% compute v_start
    v_start = 0;
    for i_start = 1:n(start)
        v_start = v_start + kron(squeeze(tt_cores{start}(:,i_start,:)), squeeze(ttX{start}(:,i_start,:)));
    end
    if start == 1
        v_start = v_start';
    end
    % now we go till stop
    for k = start+1:stop
        Vk = 0;
        for ik = 1:n(k)
            Vk = Vk + kron(squeeze(tt_cores{k}(:,ik,:)), squeeze(ttX{k}(:,ik,:)));
        end
        v_start = v_start * Vk;
    end
    X = v_start;
    if strcmp(part,'L')
        X = reshape(X, [ranks(stop+1), ranks(stop+1)]);
    elseif strcmp(part,'R')
        X = reshape(X, [ranks(start), ranks(start)]);
    end