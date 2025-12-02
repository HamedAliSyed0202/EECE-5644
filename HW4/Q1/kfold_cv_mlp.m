function [bestHidden, cvErr] = ...
    kfold_cv_mlp(X, Y, Kfold, hidden_candidates)
%KFOld_CV_MLP  K-fold CV for 1-hidden-layer MLP using patternnet

N = size(X,1);
indices = crossvalind('Kfold', N, Kfold);

X_T = X.';                       % [features x samples]
T   = labels_to_targets(Y);      % [N x 2]
T_T = T.';                       % [2 x N]

nH = numel(hidden_candidates);
cvErr = zeros(1, nH);

for hi = 1:nH
    H = hidden_candidates(hi);
    foldErr = zeros(Kfold,1);

    for k = 1:Kfold
        valIdx   = (indices == k);
        trainIdx = ~valIdx;

        net = patternnet(H);
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;

        % divide data manually into train / "test"
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = find(trainIdx);
        net.divideParam.valInd   = [];
        net.divideParam.testInd  = find(valIdx);

        net = train(net, X_T, T_T);

        % Predictions on validation samples
        Yhat = net(X_T(:,valIdx));
        [~, idxMax] = max(Yhat,[],1);

        pred = ones(numel(idxMax),1);
        pred(idxMax == 1) = -1;   % class 1 → -1, class 2 → +1

        foldErr(k) = mean(pred ~= Y(valIdx));
    end

    cvErr(hi) = mean(foldErr);
end

[~, bestIdx] = min(cvErr);
bestHidden = hidden_candidates(bestIdx);

end
