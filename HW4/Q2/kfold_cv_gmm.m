function [bestK, meanValLogL] = kfold_cv_gmm(X, Kfold, K_candidates)
%KFOld_CV_GMM  K-fold cross-validation for GMM
%   X: [N x D] normalized feature matrix
%   K_candidates: vector of candidate numbers of components
%
%   Returns:
%     bestK         - selected K maximizing mean validation log-likelihood
%     meanValLogL   - array of mean validation log-likelihoods per K

N = size(X,1);
indices = crossvalind('Kfold', N, Kfold);

nK = numel(K_candidates);
meanValLogL = zeros(1, nK);

options = statset('MaxIter', 300, 'Display', 'off');

for ki = 1:nK
    K = K_candidates(ki);
    foldLogL = zeros(Kfold,1);

    for f = 1:Kfold
        valIdx   = (indices == f);
        trainIdx = ~valIdx;

        Xtrain = X(trainIdx,:);
        Xval   = X(valIdx,:);

        % Fit GMM on training fold
        gm = fitgmdist(Xtrain, K, ...
            'RegularizationValue', 1e-6, ...
            'Options', options, ...
            'Replicates', 3);

        % Compute average log-likelihood on validation fold
        logpdfVals = log(pdf(gm, Xval));   % may contain -Inf, okay for mean
        foldLogL(f) = mean(logpdfVals);
    end

    meanValLogL(ki) = mean(foldLogL);
end

% Choose K that maximizes validation log-likelihood
[~, idxBest] = max(meanValLogL);
bestK = K_candidates(idxBest);

end
