function [X, Y] = generate_ring_data(N, r_minus, r_plus, sigma)
%GENERATE_RING_DATA  Generate points on two noisy concentric rings
%   Class -1  : radius r_minus
%   Class +1  : radius r_plus
%
%   N is total number of samples (assumed even).

if mod(N,2) ~= 0
    error('N must be even so that N/2 samples per class.');
end

N_per = N/2;

% Class -1 (inner ring)
theta1 = -pi + 2*pi*rand(N_per,1);         % uniform[-pi,pi]
n1     = sigma * randn(N_per,2);          % N(0, sigma^2 I)
base1  = [cos(theta1), sin(theta1)];
X1     = r_minus * base1 + n1;
Y1     = -ones(N_per,1);

% Class +1 (outer ring)
theta2 = -pi + 2*pi*rand(N_per,1);
n2     = sigma * randn(N_per,2);
base2  = [cos(theta2), sin(theta2)];
X2     = r_plus * base2 + n2;
Y2     =  ones(N_per,1);

% Combine
X = [X1; X2];
Y = [Y1; Y2];

end
