% setting_total_variation - set up a total variation problem
%
%   Copyright (c) 2014 Charles Deledalle

%%% Build signal f
f0 = double(imread('flinstones.png'));
[n1, n2] = size(f0);
f0 = f0(:);
N = n1 * n2;

%%% Define degradations
P = N;
phi0 = op_conv(n1, n2, 2);
sig = 10;

%%% Generate the observation y
y = phi0.A(f0) + sig * randn(P, 1);

%%% Define synthesis dictionary Psi
psi = op_id(n1, n2);

%%% Define analysis dictionary D
D = op_tv(n1, n2);

%%% Define the norm and its derivatives
tv_isotrope = true;
if tv_isotrope
    J = 2;
    amplitude.energy = @(a, lambda) ...
        lambda * sum(sum(sqrt(sum(a.^2, 3))));
    amplitude.def = @(a, lambda) ...
        1/lambda * repmat(sqrt(sum(a.^2, 3)), [1 1 J]);
    amplitude.D{1} = @(a, lambda, da) ...
        1/lambda * repmat(sum(a .* da, 3) ./ sqrt(sum(a.^2, 3)), [1 1 J]);
    amplitude.D{2} = @(a, lambda) ...
        -1/lambda^2 * repmat(sqrt(sum(a.^2, 3)), [1 1 J]);

    amplitude.energy = @(a, lambda) ...
        call(vect(@(a) amplitude.energy(a, lambda), n1, n2, J), a);
    amplitude.def = @(a, lambda) ...
        call(vect(@(a) amplitude.def(a, lambda), n1, n2, J), a);
    amplitude.D{1} = @(a, lambda, da) ...
        call(vect(@(a) amplitude.D{1}(a, lambda, reshape(da, n1, n2, J)), n1, n2, J), a);
    amplitude.D{2} = @(a, lambda) ...
        call(vect(@(a) amplitude.D{2}(a, lambda), n1, n2, J), a);
else
    amplitude.energy  = @(a, lambda)     lambda * sum(abs(a));
    amplitude.def     = @(a, lambda)     1/lambda * abs(a);
    amplitude.D{1}    = @(a, lambda, da) 1/lambda * sign(a) .* da;
    amplitude.D{2}    = @(a, lambda)     -1/lambda^2 * abs(a);
end

%%% Define solver
stop_func = @(it, energy, update) ...
    it >= 100;

solver_for_mc = @(y, lambda, delta) ...
    sparse_analysis_gfb(y, lambda, ...
                        phi0, psi, D, ...
                        amplitude, ...
                        stop_func, ...
                        'AppliedJacY', ...
                        delta);

solver_for_fdmc = @(y, lambda) ...
    sparse_analysis_gfb(y, lambda, ...
                        phi0, psi, D, ...
                        amplitude, ...
                        stop_func, ...
                        'JacTheta');

%%% Parameters
param.show = @(lambda) fprintf('%.8f', lambda);
param.ok = @(lambda) (min(lambda, [], 1) > 0);
param.init = ...
    1/2 * sig^2 * P / ...
    amplitude.energy(D.AS(phi0.ML(y)), 1) / 2;

%%% Risk conf
risk.type = 'projection';
risk.delta = randn(size(y));
