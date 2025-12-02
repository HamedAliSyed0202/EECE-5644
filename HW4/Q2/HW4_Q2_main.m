%% EECE5644 – Assignment 4 – Question 2
% GMM-based clustering to segment a color image

clear; close all; clc;
rng(1);

%% ------------------------------------------------------------------------
% 1) Load a BSDS300 color image
%% ------------------------------------------------------------------------
% Put any BSDS300 image in this folder and set its name here, e.g. '42049.jpg'
imgFilename = '02.jpg';   % <-- CHANGE THIS

I = imread(imgFilename);
I = im2double(I);      % convert to [0,1]
[H, W, C] = size(I);

if C ~= 3
    error('Image must be RGB color (3 channels).');
end

fprintf('Loaded image %s of size %d x %d\n', imgFilename, H, W);

%% ------------------------------------------------------------------------
% 2) Build 5-D feature vector per pixel:
%    [row_index, col_index, R, G, B]
%% ------------------------------------------------------------------------
[colGrid, rowGrid] = meshgrid(1:W, 1:H);   % row (y), col (x)

row_feat = rowGrid(:);
col_feat = colGrid(:);

R = I(:,:,1); G = I(:,:,2); B = I(:,:,3);
R = R(:); G = G(:); B = B(:);

X = [row_feat, col_feat, R, G, B];   % N x 5
N = size(X,1);

% Normalize each feature dimension individually to [0,1]
X_norm = zeros(size(X));
for d = 1:size(X,2)
    xmin = min(X(:,d));
    xmax = max(X(:,d));
    if xmax > xmin
        X_norm(:,d) = (X(:,d) - xmin) / (xmax - xmin);
    else
        X_norm(:,d) = 0;   % constant feature (unlikely but safe)
    end
end

%% ------------------------------------------------------------------------
% 3) K-fold CV to select best number of GMM components
%% ------------------------------------------------------------------------
Kfold = 5;
K_candidates = 2:6;    % try 2–6 clusters

% To speed up CV for large images, optionally subsample points
maxPoints = 80000;
if N > maxPoints
    fprintf('Downsampling from %d to %d pixels for CV...\n', N, maxPoints);
    idxSub = randperm(N, maxPoints);
    X_cv = X_norm(idxSub,:);
else
    X_cv = X_norm;
end

[bestK, meanValLogL] = kfold_cv_gmm(X_cv, Kfold, K_candidates);

fprintf('\n===== GMM model selection (CV) =====\n');
for i = 1:numel(K_candidates)
    fprintf('K = %d: mean validation log-likelihood = %.4f\n', ...
        K_candidates(i), meanValLogL(i));
end
fprintf('Selected K* = %d components\n', bestK);

figure;
plot(K_candidates, meanValLogL, '-o','LineWidth',1.5);
xlabel('Number of components K');
ylabel('Mean validation log-likelihood');
title('GMM model selection via K-fold CV');
grid on;

%% ------------------------------------------------------------------------
% 4) Fit best GMM on ALL pixels
%% ------------------------------------------------------------------------
options = statset('MaxIter', 500, 'Display', 'final');

gmBest = fitgmdist(X_norm, bestK, ...
    'RegularizationValue', 1e-6, ...
    'Options', options, ...
    'Replicates', 3);

% Posterior probabilities and hard labels
P = posterior(gmBest, X_norm);      % N x bestK
[~, labels] = max(P, [], 2);        % N x 1, in {1,...,bestK}

labelImg = reshape(labels, [H, W]);

%% ------------------------------------------------------------------------
% 5) Build color segmentation image using mean RGB of each cluster
%% ------------------------------------------------------------------------
clusterColors = zeros(bestK, 3);
for k = 1:bestK
    mask = (labels == k);
    if any(mask)
        clusterColors(k,1) = mean(R(mask));
        clusterColors(k,2) = mean(G(mask));
        clusterColors(k,3) = mean(B(mask));
    else
        clusterColors(k,:) = [0 0 0];
    end
end

segImg = zeros(H,W,3);
for k = 1:bestK
    mask = (labelImg == k);
    for c = 1:3
        tmp = segImg(:,:,c);
        tmp(mask) = clusterColors(k,c);
        segImg(:,:,c) = tmp;
    end
end

%% ------------------------------------------------------------------------
% 6) Display original image, segmentation, and label map
%% ------------------------------------------------------------------------
figure;
subplot(1,3,1);
imshow(I);
title('Original image');

subplot(1,3,2);
imshow(segImg);
title(sprintf('GMM segmentation (K = %d)', bestK));

subplot(1,3,3);
imagesc(labelImg);
axis image off;
title('Label map (cluster indices)');
colormap('jet'); colorbar;
