% gen_fig52b - produces Figure 5.2 (b) of the SUGAR paper
%
%   Copyright (c) 2014 Charles Deledalle

clear all
close all

addpathrec('.');
state = deterministic('on');

%%% Load problem settings
setting_total_variation;

%%% Optimize SURE-FDMC with quasi-Newton
lambda_ini = param.init;
objective = @(lambda) estimate_risk_fdmc(y, lambda, ...
                                         param, phi0, sig, ...
                                         solver_for_fdmc, risk, ...
                                         'sugar');
fprintf('\nStart BFGS\n');
tic;
[lambda_bfgs_opt, ~, ~, N_rec, ...
 lambda_bfgs_rec, asure_bfgs_rec, asugar_bfgs_rec] = ...
    perform_bfgs(objective, lambda_ini);
time_bfgs = toc;

fprintf('\nGet the final image\n');
f_bfgs_opt = solver_for_fdmc(y, lambda_bfgs_opt);

%%% Define range of lambdas
margin = (max(lambda_bfgs_rec) - min(lambda_bfgs_rec)) / 10;
if margin > 0
    a = max(abs(lambda_bfgs_rec - lambda_bfgs_opt));
    lmin = max(lambda_bfgs_opt - a - margin, ...
               min(lambda_bfgs_rec)/2);
    lmax = lambda_bfgs_opt + a + margin;
else
    lmin = min(lambda_bfgs_rec) * 0.5;
    lmax = min(lambda_bfgs_rec) * 1.5;
end
lnb = 21;
lambda_list_bg = linspace(lmin, lmax, lnb);
lambda_list_bg = sort([lambda_list_bg lambda_bfgs_rec]);

%%% Estimate SURE-FDMC with Exhaustive search
fn_backup = 'data/figure_2_b_background.mat';
if exist(fn_backup)
    fprintf('\nLoad background\n');
    load(fn_backup);
else
    fprintf('\nCompute backup by exhaustive search (may take some time)\n');
    tic;
    [asure, f] = estimate_risk_fdmc(y, lambda_list_bg, ...
                                    param, phi0, sig, ...
                                    solver_for_fdmc, risk);
    time_exhaustive = toc;

    % Compute Squared Errors
    for k = 1:length(lambda_list_bg)
        se(k) = norm(f{k} - f0)^2 / P;
        ase(k) = norm(phi0.Pi(f{k} - f0))^2 / (sig^2 * phi0.AAS_PseudoInv_trace);
    end

    save(fn_backup, 'asure', 'se', 'ase', 'time_exhaustive');
end
[~, k_opt] = min(asure);

%%% Show stats
fprintf('\nShow results\n');
fprintf('  Original PSNR: %.2f\n', ...
        psnr(phi0.ML(y), f0));
fprintf('  Final PSNR: %.2f\n', ...
        psnr(f_bfgs_opt, f0));
fprintf('  Time BFGS: %.2f\n', time_bfgs);
fprintf('  Time Exhaustive: %.2f\n', time_exhaustive);


%%% Draw Risk curves
ratio = 16/9;
scaling = ...
    (max(lambda_list_bg) - min(lambda_list_bg)) / ...
    (max(asure) - min(asure)) / ...
    ratio;

figure,
a = abs(asure(floor(3*end/4)) - asure(floor(end/2))) ...
    / abs(se(floor(3*end/4)) - se(floor(end/2)));
b = a * se(k_opt) - asure(k_opt);
p1 = plot(lambda_list_bg / scaling, a*se-b, 'r-.');
hold on
t = mean(ase - asure);
p2 = plot(lambda_list_bg / scaling, ase-t, 'k--', 'LineWidth', 1);
p3 = plot(lambda_list_bg / scaling, asure, 'k-', 'LineWidth', 1);
axis image
xlim([min(lambda_list_bg) max(lambda_list_bg)]/scaling);
m = (max(asure) - min(asure)) / 10;
ylim([min([asure a*se-b])-m max([asure a*se-b])+m]);
hold on
h = (lmax-lmin)/scaling/2000;
for k0 = 1:length(lambda_bfgs_rec)
    l0 = lambda_bfgs_rec(k0) / scaling;
    r0 = asure_bfgs_rec(k0);
    fref = min(asure_bfgs_rec);
    g0 = asugar_bfgs_rec(1, k0) * scaling;
    alpha = 100;
    h0 = -h * sign(g0) / norm([1 g0]);
    h1 = h/3 / norm([1 abs(g0)]);
    p4 = plot(l0 + [-h0 h0] * alpha + h1*200 * g0, ...
              r0 + [-h0 h0] * g0 * alpha - h1*200, ...
              'o-', ...
              'Color', [191 0 191]/255, 'LineWidth', 2, ...
              'MarkerFaceColor', [191 0 191]/255, ...
              'MarkerEdgeColor', [191 0 191]/255, ...
              'MarkerSize', 4);
    switch k0
      case 1
        p5 = plot(l0, r0, 'xg', 'LineWidth', 2, 'MarkerSize',10);
      case length(lambda_bfgs_rec)
        p7 = plot(l0, r0, 'xr', 'LineWidth', 2, 'MarkerSize',10);
      otherwise
        p6 = plot(l0, r0, 'xb', 'LineWidth', 2, 'MarkerSize',10);
    end
    text(l0 - h1*400 * g0, r0 + h1*400, ...
         num2str(k0), 'FontWeight', 'bold');
end
xlabel('Regularization parameter \lambda');
ylabel('Quadratic cost');
step_tick = (lmax-lmin)/5;
lambda_tick = (lmin+step_tick):step_tick:(lmax-step_tick);
set(gca, 'XTick', lambda_tick / scaling);
set(gca, 'XTickLabel', ...
         mycellfun(@(f) sprintf('%.3f', f), ...
                   num2cell(lambda_tick)));
legend([p1 p2 p3 p4 p5 p6 p7], ...
       'Risk', ...
       'Projection risk', ...
       'GSURE_{FDMC}', ...
       'GSUGAR_{FDMC} based slope', ...
       'BFGS initialization', ...
       'BFGS iterates', ...
       'BFGS final');

legend([p3 p4 p5 p6 p7], ...
       'GSURE_{FDMC}', ...
       'GSUGAR_{FDMC} based slope', ...
       'BFGS initialization', ...
       'BFGS iterates', ...
       'BFGS final');

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
