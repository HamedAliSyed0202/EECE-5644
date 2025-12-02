function plot_decision_boundary(predictFcn, X, Y)
%PLOT_DECISION_BOUNDARY  Visualize classifier decision boundary in 2D
%   predictFcn: function handle @(X) -> predicted labels (-1/+1)
%   X: [N x 2], Y: [N x 1]

x1min = min(X(:,1)) - 1;
x1max = max(X(:,1)) + 1;
x2min = min(X(:,2)) - 1;
x2max = max(X(:,2)) + 1;

[x1Grid, x2Grid] = meshgrid(linspace(x1min,x1max,200), ...
                            linspace(x2min,x2max,200));
Xgrid = [x1Grid(:), x2Grid(:)];

Ygrid = predictFcn(Xgrid);
Ygrid = reshape(Ygrid, size(x1Grid));

contourf(x1Grid, x2Grid, Ygrid, [-1 0 1], 'LineColor','none');
hold on;
colormap([1 0.8 0.8; 0.8 0.8 1]); % light red / light blue

scatter(X(Y==-1,1), X(Y==-1,2), 25, 'r', 'filled');
scatter(X(Y==+1,1), X(Y==+1,2), 25, 'b', 'filled');

axis equal; grid on;
xlabel('x_1'); ylabel('x_2');
legend({'Decision region','Class -1','Class +1'}, 'Location','bestoutside');

end
