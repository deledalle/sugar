% setting_multiscale - set up a multiscale wavelet
%                      problem with one parameter
%                      per scale
%
%   Copyright (c) 2014 Charles Deledalle

%%% Build signal f
if ~exist('f0', 'var')
    f0 = double(imread('cameraman.png'));
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
amplitude.energy = @(a, lambdas)    sum(sum(abs(a) * diag(lambdas)));
for j = 1:J
    amplitude.subenergy{j} = @(a, lambda)    sum(sum(abs(a(:,j)) * lambda));
end
amplitude.def = @(a, lambdas)       abs(a) * diag(1./lambdas);
amplitude.D{1} = @(a, lambdas, da)  (sign(a) .* da) * diag(1./lambdas);
M = zeros(J, J^2);
for j = 1:J
    M(j, (J+1)*j - J) = 1;
end
amplitude.D{2} = @(a, lambdas)      abs(a) * -diag(1./lambdas.^2) * M;

amplitude.energy = @(a, lambdas) ...
    call(vect(@(a) amplitude.energy(a, lambdas), 2*N, J), a);
for j = 1:J
    amplitude.subenergy{j} = @(a, lambda) ...
        call(vect(@(a) amplitude.subenergy{j}(a, lambda), 2*N, J), a);
end
amplitude.def = @(a, lambdas) ...
    call(vect(@(a) amplitude.def(a, lambdas), 2*N, J), a);
amplitude.D{1} = @(a, lambdas, da) ...
    call(vect(@(da) amplitude.D{1}(reshape(a, 2*N, J), lambdas, da), 2*N, J), da);
amplitude.D{2} = @(a, lambdas) ...
    reshape(call(vect(@(a) amplitude.D{2}(a, lambdas), 2*N, J), a), ...
            2*N*J, J);

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
param.init = zeros(J, 1);
param.init(:) = ...
    1/2 * sig^2 * P ./ ...
    amplitude.energy(D.AS(phi0.ML(y)), 1) / 2;


%%% Risk conf
risk.type = 'projection';
risk.delta = randn(size(y));
