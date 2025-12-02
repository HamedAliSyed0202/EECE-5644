%% EECE5644 – Assignment 4 – Question 1
clear; close all; clc;
rng(1);

%% ------------------------------------------------------------------------
% 1) Generate data (two noisy rings)
%% ------------------------------------------------------------------------
Ntrain = 1000;
Ntest  = 10000;

r1 = 2; 
r2 = 4;
sigma = 1;

[Xtrain, Ytrain] = generate_ring_data(Ntrain, r1, r2, sigma);
[Xtest,  Ytest ] = generate_ring_data(Ntest,  r1, r2, sigma);

figure; gscatter(Xtrain(:,1), Xtrain(:,2), Ytrain);
axis equal; grid on; title('Training Data');

%% ------------------------------------------------------------------------
% 2) SVM (Gaussian kernel) – K-fold CV
%% ------------------------------------------------------------------------
K = 5;
C_values     = logspace(-2,2,5);
sigma_values = [0.2 0.5 1 2 5];

N = length(Ytrain);
indices = crossvalind('Kfold', N, K);

cvErr = zeros(length(sigma_values), length(C_values));

for si = 1:length(sigma_values)
    sig = sigma_values(si);

    for ci = 1:length(C_values)
        C = C_values(ci);

        fold_err = zeros(K,1);

        for k = 1:K
            vlIdx = (indices == k);
            trIdx = ~vlIdx;

            mdl = fitcsvm(Xtrain(trIdx,:), Ytrain(trIdx), ...
                'KernelFunction','rbf', ...
                'KernelScale', sig, ...
                'BoxConstraint', C, ...
                'Standardize',true);

            pred = predict(mdl, Xtrain(vlIdx,:));
            pred = pred(:);                 % ensure column
            ytrue = Ytrain(vlIdx);
            ytrue = ytrue(:);               % ensure column

            fold_err(k) = mean(pred ~= ytrue);
        end

        cvErr(si,ci) = mean(fold_err);
    end
end

% Best hyperparameters
[minError, idxMin] = min(cvErr(:));
[best_si, best_ci] = ind2sub(size(cvErr), idxMin);

bestSigma = sigma_values(best_si);
bestC     = C_values(best_ci);

fprintf('\n===== SVM Results =====\n');
fprintf('Best sigma = %.4f\n', bestSigma);
fprintf('Best C     = %.4f\n', bestC);
fprintf('CV error   = %.4f\n', minError);

% Train final SVM
svmModel = fitcsvm(Xtrain, Ytrain, ...
    'KernelFunction','rbf', ...
    'KernelScale', bestSigma, ...
    'BoxConstraint', bestC, ...
    'Standardize',true);

Ypred = predict(svmModel, Xtest);
testErr_svm = mean(Ypred ~= Ytest);

fprintf('Test Error (SVM) = %.4f\n', testErr_svm);

% Plot SVM decision
figure;
plot_decision_boundary(@(X) predict(svmModel,X), Xtrain, Ytrain);
title(sprintf('SVM Decision Boundary (Test Err = %.3f)', testErr_svm));


%% ------------------------------------------------------------------------
% 3) MLP (1 hidden layer) – K-fold CV
%% ------------------------------------------------------------------------
hidden_list = [5 10 20 40];
cvErr_mlp = zeros(size(hidden_list));

X_T = Xtrain.';                           % [2 x N]
T = labels_to_targets(Ytrain).';          % [2 x N]

for hi = 1:length(hidden_list)
    h = hidden_list(hi);

    foldErr = zeros(K,1);

    for k = 1:K
        vlIdx = (indices == k);
        trIdx = ~vlIdx;

        net = patternnet(h);
        net.trainParam.showWindow = false;
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = find(trIdx);
        net.divideParam.valInd   = [];
        net.divideParam.testInd  = find(vlIdx);

        net = train(net, X_T, T);

        % Predict on validation fold
        yhat = net(X_T(:,vlIdx));
        [~, idxMax] = max(yhat,[],1);

        pred = ones(length(idxMax),1);
        pred(idxMax == 1) = -1;   % convert to {-1,+1}

        ytrue = Ytrain(vlIdx);
        ytrue = ytrue(:);

        foldErr(k) = mean(pred ~= ytrue);
    end

    cvErr_mlp(hi) = mean(foldErr);
end

% Best hidden units
[~, idxH] = min(cvErr_mlp);
bestHidden = hidden_list(idxH);

fprintf('\n===== MLP Results =====\n');
fprintf('Best hidden units = %d\n', bestHidden);

% Train final MLP
netFinal = patternnet(bestHidden);
netFinal.trainParam.showWindow = false;
netFinal.divideFcn = 'dividetrain';
netFinal = train(netFinal, X_T, T);

% Test prediction
Ypred_mlp = mlp_predict_labels(netFinal, Xtest);
testErr_mlp = mean(Ypred_mlp ~= Ytest);

fprintf('Test Error (MLP) = %.4f\n', testErr_mlp);

figure;
plot_decision_boundary(@(X) mlp_predict_labels(netFinal,X), Xtrain, Ytrain);
title(sprintf('MLP Decision (Hidden=%d, Test Err=%.3f)', bestHidden, testErr_mlp));
