% setting_matrix_completion - set up a random matrix completion problem
%
%   Copyright (c) 2014 Charles Deledalle


%%% Parameters
n1 = 1000;
n2 = 100;
alpha = 1;

%%% Build signal f
n = min(n1, n2);
N = n1*n2;

[U,r] = qr(randn(n1));
[V,r] = qr(randn(n2));
S = (1:n).^(-alpha);
Delta = zeros(n1, n2);
Delta(1:n, 1:n) = diag(S);
f0 = U*Delta*V';
f0 = f0(:);

%%% Define degradations
P = floor(N/4);
phi0 = op_random_masking(n1, n2, P);
sig = std(f0(:))/2;

%%% Generate the observation y
y = phi0.A(f0) + sig * randn(P, 1);

%%% Define solver
stop_func = @(it, energy, update) ...
    it >= 100;

solver_for_mc = @(y, lambda, delta) ...
    nuclearnorm_fb(y, lambda, ...
                   phi0, n1, n2, ...
                   stop_func, ...
                   'AppliedJacY', ...
                   delta);

solver_for_fdmc = @(y, lambda) ...
    nuclearnorm_fb(y, lambda, ...
                   phi0, n1, n2, ...
                   stop_func, ...
                   'JacTheta');

%%% Parameters
param.ok = @(lambda) min(lambda, [], 1) > 0;
param.show = @(lambda) fprintf('%.8f', lambda);
param.init = ...
    1/2 * sig^2 * P / ...
    sum(abs(svd(phi0.ML(y)))) / 2;

%%% Risk conf
risk.type = 'prediction';
risk.delta = randn(size(y));
