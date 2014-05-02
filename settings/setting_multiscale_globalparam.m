% setting_multiscale_globalparameter - set up a multiscale wavelet
%                                      problem with one global parameter
%
%   Copyright (c) 2014 Charles Deledalle

%%% Build signal f
if ~exist('f0', 'var')
    f0 = double(imread('house.png'));
end
[n1, n2] = size(f0);
f0 = f0(:);
N = n1 * n2;

%%% Define degradations
P = N/2;
phi0 = op_cs(n1, n2, P);
sig = 10;

%%% Generate the observation y
y = phi0.A(f0) + sig * randn(P, 1);

%%% Define synthesis dictionary Psi
psi = op_id(n1, n2);

%%% Define analysis dictionary D
if ~exist('J', 'var')
    J = 3;
end
D = op_daub4_udwt_analysis(n1, n2, 1:J, 0);

%%% Define the norm and its derivative
amplitude.energy = @(a, lambda)    sum(sum(abs(a) * lambda));
amplitude.def = @(a, lambda)       abs(a) * 1/lambda;
amplitude.D{1} = @(a, lambda, da)  sign(a) .* da * 1/lambda;
amplitude.D{2} = @(a, lambda)      abs(a) * -1/lambda.^2;

%%% Define solver
stop_func = @(it, energy, update) ...
    it >= 100;

solver_for_mc = @(y, lambda, delta) ...
    sparse_analysis_cp(y, lambda, ...
                       phi0, psi, D, ...
                       amplitude, ...
                       stop_func, ...
                       'AppliedJacY', ...
                       delta);

solver_for_fdmc = @(y, lambda, delta) ...
    sparse_analysis_cp(y, lambda, ...
                       phi0, psi, D, ...
                       amplitude, ...
                       stop_func, ...
                       'JacTheta');

%%% Parameters
param.show = @(lambda) fprintf('%.8f ', lambda);
param.ok = @(lambda) (min(lambda, [], 1) > 0);
param.init = ...
    1/2 * sig^2 * P ./ ...
    amplitude.energy(D.AS(phi0.ML(y)), 1) / 2;

%%% Risk conf
risk.type = 'projection';
risk.delta = randn(size(y));
