% example - produces an example of the SUGAR approach
%
%   Solve a deconvolution problem
%   with Total Variation
%   using Chambolle Pock algorithm
%
%   The near-optimal parameter is obtained with SUGAR and BFGS
%
%   Copyright (c) 2014 Charles Deledalle

clear all
close all

addpathrec('.');
state = deterministic('on');

%%% Build signal f
f0 = double(imread('peppers.png'));
[n1, n2] = size(f0);
f0 = f0(:);
N = n1 * n2;

%%% Define degradations
phi0 = op_conv(n1, n2, 2);
P = N;
sig = 10;

%%% Generate the observation y
y = phi0.A(f0) + sig * randn(P, 1);

%%% Define synthesis/analysis dictionaries
psi = op_id(n1, n2);
D = op_tv(n1, n2);

%%% Define the l1-norm and its derivatives
amplitude.energy  = @(a, lambda)     lambda * sum(abs(a));
amplitude.def     = @(a, lambda)     1/lambda * abs(a);
amplitude.D{1}    = @(a, lambda, da) 1/lambda * sign(a) .* da;
amplitude.D{2}    = @(a, lambda)     -1/lambda^2 * abs(a);

%%% Define solver (Chambolle-Pock)
stop_func = @(it, energy, update) ...
    it >= 100;
solver_for_fdmc = @(y, lambda) ...
    sparse_analysis_cp(y, lambda, ...
                       phi0, psi, D, ...
                       amplitude, ...
                       stop_func, ...
                       'JacTheta');

%%% Define parameter tools
param.show = @(lambda) fprintf('%.8f', lambda);
param.ok = @(lambda) (min(lambda, [], 1) > 0);
param.init = 1.5 * sig;

%%% Optimize SURE-FDMC with quasi-Newton
lambda_ini = param.init;
risk.type = 'prediction';
risk.delta = randn(P, 1);
objective = @(lambda) estimate_risk_fdmc(y, lambda, ...
                                         param, phi0, sig, ...
                                         solver_for_fdmc, ...
                                         risk, ...
                                         'sugar');

fprintf('\nStart BFGS\n');
lambda_bfgs_opt = perform_bfgs(objective, lambda_ini);

fprintf('\nGet the final image\n');
f_bfgs_opt      = solver_for_fdmc(y, lambda_bfgs_opt);

%%% Show stats
fprintf('\nShow results\n');
fprintf('  Original PSNR: %.2f\n', psnr(phi0.ML(y), f0));
fprintf('  Final PSNR: %.2f\n', psnr(f_bfgs_opt, f0));

%%% Plot images
figure
subplot(2,2,1)
plotimage(reshape(f0, n1, n2));
title('Original');
subplot(2,2,2)
plotimage(reshape(y, n1, n2));
title(sprintf('Observed PSNR: %f', psnr(y, f0)));
subplot(2,2,3)
plotimage(reshape(phi0.ML(y), n1, n2));
title(sprintf('Least-square PSNR: %f', psnr(phi0.ML(y), f0)));
subplot(2,2,4)
plotimage(reshape(f_bfgs_opt, n1, n2));
title(sprintf('Restored PSNR: %f', psnr(f_bfgs_opt, f0)));
linkaxes

deterministic('off', state);