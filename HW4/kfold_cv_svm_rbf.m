function [bestC, bestSigma, cvErr, C_grid, S_grid] = ...
    kfold_cv_svm_rbf(X, Y, Kfold, C_values, sigma_values)
%KFOld_CV_SVM_RBF  K-fold CV for SVM with Gaussian kernel

N = size(X,1);
indices = crossvalind('Kfold', N, Kfold);

nC = numel(C_values);
nS = numel(sigma_values);
cvErr = zeros(nS, nC);  % rows: sigma, cols: C

for si = 1:nS
    for ci = 1:nC
        Cval = C_values(ci);
        sig  = sigma_values(si);
        foldErr = zeros(Kfold,1);

        for k = 1:Kfold
            testIdx  = (indices == k);
            trainIdx = ~testIdx;

            mdl = fitcsvm(X(trainIdx,:), Y(trainIdx), ...
                'KernelFunction','rbf', ...
                'KernelScale', sig, ...
                'BoxConstraint', Cval, ...
                'Standardize', true);

            Yval = predict(mdl, X(testIdx,:));
            foldErr(k) = mean(Yval ~= Y(testIdx));
        end

        cvErr(si,ci) = mean(foldErr);
    end
end

% Get best (C, sigma)
[minErr, idx] = min(cvErr(:));
[bestSi, bestCi] = ind2sub(size(cvErr), idx);
bestC     = C_values(bestCi);
bestSigma = sigma_values(bestSi);

[C_grid, S_grid] = meshgrid(C_values, sigma_values);
fprintf('\nSVM: minimum CV error = %.4f\n', minErr);

end
