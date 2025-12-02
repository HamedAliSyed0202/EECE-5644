function Ypred = mlp_predict_labels(net, X)
%MLP_PREDICT_LABELS  Predict class labels (-1,+1) using trained net
%   X : [N x 2] data matrix

X_T = X.';               % [2 x N]
Yhat = net(X_T);         % [2 x N]
[~, idxMax] = max(Yhat, [], 1);

Ypred = ones(size(idxMax(:)));
Ypred(idxMax(:) == 1) = -1;   % first output neuron → class -1

end
