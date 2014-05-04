% gen_fig52a - produces Figure 5.2 (a) of the SUGAR paper
%
%   Copyright (c) 2014 Charles Deledalle

clear all
close all

addpathrec('.');
state = deterministic('on');

%%% Load problem settings
setting_total_variation;

%%% Define range of lambdas
lmin = 0.70536740;
lmax = 1.55589575;
lnb = 52;
lambda_list = linspace(lmin, lmax, lnb);

%%% Estimate SURE-MC with Exhaustive search
fprintf('\nStart exhaustive search (%d points, may take some time)\n', lnb);
[asure, f] = estimate_risk_mc(y, lambda_list, ...
                              param, phi0, sig, ...
                              solver_for_mc, risk);
[~, k_opt] = min(asure);
lambda_opt = lambda_list(k_opt);

%%% Compute Squared Errors
for k = 1:length(lambda_list)
    se(k) = norm(f{k} - f0)^2 / P;
    ase(k) = norm(phi0.Pi(f{k} - f0))^2 / (sig^2 * phi0.AAS_PseudoInv_trace);
end

%%% Draw Risk curves
ratio = 16/9;
scaling = ...
    (max(lambda_list) - min(lambda_list)) / ...
    (max(asure) - min(asure)) / ...
    ratio;

a = abs(asure(floor(3*end/4)) - asure(floor(end/2))) ...
    / abs(se(floor(3*end/4)) - se(floor(end/2)));
b = a * se(k_opt) - asure(k_opt);
t = mean(ase - asure);

figure,
plot(lambda_list / scaling, a*se-b, ...
     'r-.');
hold on
plot(lambda_list / scaling, ase-t, ...
     'k--', 'LineWidth', 1);
plot(lambda_list / scaling, asure, ...
     'k-', 'LineWidth', 1);
plot(lambda_list((4+2-mod(k_opt,2)):4:(end-2)) / scaling, asure((4+2-mod(k_opt,2)):4:(end-2)), ...
     'xb', 'LineWidth', 2, 'MarkerSize', 10);
plot(lambda_list(k_opt) / scaling, asure(k_opt), ...
     'xr', 'LineWidth', 2, 'MarkerSize', 10);
axis image
xlim([min(lambda_list) max(lambda_list)] / scaling);
m = (max(asure) - min(asure)) / 10;
ylim([min([asure a*se-b])-m max([asure a*se-b])+m]);
step_tick = (lmax-lmin)/5;
lambda_tick = (lmin+step_tick):step_tick:(lmax-step_tick);
set(gca, 'XTick', lambda_tick / scaling);
set(gca, 'XTickLabel', ...
         mycellfun(@(f)  sprintf('%.3f', f), ...
                   num2cell(lambda_tick)));
xlabel('Regularization parameter \lambda');
ylabel('Quadratic cost');
legend('Risk', 'Projection risk', 'GSURE_{MC}', ...
       'Exh. search iterates', ...
       'Exh. search final');

%%% Plot images
figure
subplot(1,3,1)
plotimage(reshape(f0, n1, n2));
title('Original');
subplot(1,3,2)
plotimage(reshape(y, n1, n2));
title(sprintf('Observed PSNR: %f', psnr(y, f0)));
subplot(1,3,3)
plotimage(reshape(f_bfgs_opt, n1, n2));
title(sprintf('Restored PSNR: %f', psnr(f_bfgs_opt, f0)));
linkaxes

deterministic('off', state);