function T = labels_to_targets(Y)
%LABELS_TO_TARGETS  Convert labels -1/+1 → one-hot targets [N x 2]
% class -1 : [1 0]
% class +1 : [0 1]

N = numel(Y);
T = zeros(N,2);

T(Y == -1, 1) = 1;
T(Y == -1, 2) = 0;

T(Y == +1, 1) = 0;
T(Y == +1, 2) = 1;

end
