T = zeros(3,3,2,2);

G1 = reshape([1 1 1 1 1 1], [1 3 2]);        % (r1=1, n1=3, r2=2)
G2 = reshape([1 4 2 5 3 6 2 1 4 3 6 5], [2 3 2]);  % (r2=2, n2=3, r3=2)
G3 = reshape([1 0 0 1 0 1 1 0], [2 2 2]);    % (r3=2, n3=2, r4=2)
G4 = reshape([2 6 4 8], [2 2 1]);            % (r4=2, n4=2, r5=1)


for i1 = 1:3
    for i2 = 1:3
        for i3 = 1:2
            for i4 = 1:2
                % Extract TT cores
                g1 = squeeze(G1(:,i1,:)); % 1x2
                g2 = squeeze(G2(:,i2,:)); % 2x2
                g3 = squeeze(G3(:,i3,:)); % 2x2
                g4 = squeeze(G4(:,i4,:)); % 2x1

                % Multiply all cores along ranks
                val = g1' * g2 * g3 * g4; % scalar
                T(i1,i2,i3,i4) = val;
            end
        end
    end
end
