function KR = khatriRao(cores, j, d, R)
    % This is a helper functino that computes Khatri-Rao product of all but
    % j-th core in cores cell
    KR = ones(1,R);
    for k=setdiff(1:d,j)
        I = size(KR,1);
        J = size(cores{k},1);
        KR_new = zeros(I*J,R);
        for r=1:R
            KR_new(:,r) = kron(cores{k}(:,r), KR(:,r));
        end
        KR = KR_new;
    end
end