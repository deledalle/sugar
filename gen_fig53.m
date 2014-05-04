% gen_fig53 - produces the 3rd line column (e) of Figure
%             5.3 of the SUGAR paper
%
%   Copyright (c) 2014 Charles Deledalle

clear all
close all

addpathrec('.');
state = deterministic('on');

%%% Load problem settings
setting_multiscale;

%%% Optimize SURE-FDMC with quasi-Newton
lambda_ini = param.init;
objective = @(lambda) estimate_risk_fdmc(y, lambda, param, phi0, sig, ...
                                         solver_for_fdmc, risk, 'sugar');
objective_sugarfree = @(lambda) estimate_risk_fdmc(y, lambda, param, phi0, sig, ...
                                                  solver_for_fdmc, risk);

fprintf('\nStart BFGS\n');
tic;
[lambda_bfgs_opt, ~, ~, N_rec, ...
 lambda_bfgs_rec, asure_bfgs_rec, asugar_bfgs_rec] = ...
    perform_bfgs(objective, lambda_ini);
time_bfgs = toc;

fprintf('\nCompute solution\n');
f_bfgs_opt = solver_for_fdmc(y, lambda_bfgs_opt, stop_func);

%%% Show stats
fprintf('\n');
fprintf('Original PSNR: %.2f\n', psnr(phi0.ML(y), f0));
fprintf('Final PSNR: %.2f\n', psnr(f_bfgs_opt, f0));
fprintf('Time BFGS: %.2f\n', time_bfgs);

alpharange = [0.75 1 1.25];
fprintf('\nCompute for');
fprintf(' %f', alpharange);
fprintf('\n');
[asure, af] = objective_sugarfree(repmat(lambda_bfgs_opt, [1 3]) * diag(alpharange));
fprintf('\n  %f SURE/PSNR: %.2e/%.2f\n', alpharange(1), asure(1), psnr(af{1}, f0));
fprintf('\n  %f SURE/PSNR: %.2e/%.2f\n', alpharange(2), asure(2), psnr(af{2}, f0));
fprintf('\n  %f SURE/PSNR: %.2e/%.2f\n', alpharange(3), asure(3), psnr(af{3}, f0));

%%% Plot images
figure
subplot(1,3,1)
plotimage(reshape(f0, n1, n2));
title('Original');
subplot(1,3,2)
plotimage(reshape(phi0.ML(y), n1, n2));
title(sprintf('Observed PSNR: %f', psnr(phi0.ML(y), f0)));
subplot(1,3,3)
plotimage(reshape(f_bfgs_opt, n1, n2));
title(sprintf('Restored PSNR: %f', psnr(f_bfgs_opt, f0)));
linkaxes

%% Evolution of the risk
figure
plot(asure_bfgs_rec, 'r');
hold on
xlim([1 N_rec]);
xlabel('Iterations of BFGS');
ylabel('Risk evolution');

deterministic('off', state);
