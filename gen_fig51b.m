% gen_fig51b - produces Figure 5.1 (b) of the SUGAR paper
%
%   Copyright (c) 2014 Charles Deledalle

clear all
close all

addpathrec('.');
state = deterministic('on');

%%% Load problem settings
setting_matrix_completion;

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

%%% Compute background
% Define range of lambdas
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

% Compute SURE-FDMC for each lambda
fn_backup = 'data/figure_1_b_background.mat';
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
        ase(k) = norm(phi0.A(f{k} - f0))^2 / (sig^2 * P);
    end

    save(fn_backup, 'asure', 'se', 'ase', 'time_exhaustive');
end
[~, k_opt] = min(asure);

%%% Show stats
fprintf('\nShow results\n');
fprintf('  Original RMSE: %.2f\n', ...
        norm(phi0.ML(y) - f0) / norm(f0));
fprintf('  Final RMSE: %.2f\n', ...
        norm(f_bfgs_opt - f0) / norm(f0));
fprintf('  Final rank: %.2f\n', ...
        rank(reshape(f_bfgs_opt, n1, n2)));
fprintf('  Time BFGS: %.2f\n', time_bfgs);
fprintf('  Time Exhaustive: %.2f\n', time_exhaustive);


%%% Draw Risk curves
ratio = 16/9;
scaling = ...
    (max(lambda_list_bg) - min(lambda_list_bg)) / ...
    (max(asure) - min(asure)) / ...
    ratio;

figure,
a = abs(asure(floor(end/5)) - asure(floor(end/2))) ...
    / abs(se(floor(end/5)) - se(floor(end/2)));
b = a * se(k_opt) - asure(k_opt);
p1 = plot(lambda_list_bg / scaling, a*se-b, 'r-.');
hold on
t = mean(ase - asure);
p2 = plot(lambda_list_bg / scaling, ase-t, 'k--', 'LineWidth', 1);
p3 = plot(lambda_list_bg / scaling, asure, 'k-', 'LineWidth', 1);
axis image
xlim([min(lambda_list_bg) max(lambda_list_bg)]/scaling);
m = (max(asure) - min(asure)) / 10;
ylim([min(asure)-m max(asure)+m]);
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
       'Prediction risk', ...
       'SURE_{FDMC}', ...
       'SUGAR_{FDMC} based slope', ...
       'BFGS initialization', ...
       'BFGS iterates', ...
       'BFGS final');

legend([p3 p4 p5 p6 p7], ...
       'SURE_{FDMC}', ...
       'SUGAR_{FDMC} based slope', ...
       'BFGS initialization', ...
       'BFGS iterates', ...
       'BFGS final');

%%% Draw spectrums
figure,
semilogx(svd(reshape(f0, n1, n2)), 'r', 'LineWidth', 2);
hold on
plot(svd(reshape(f_bfgs_opt, n1, n2)), 'b', 'LineWidth', 2);
plot(svd(reshape(phi0.ML(y), n1, n2)), 'g--', 'LineWidth', 2);
legend('True spectrum', 'Optimal estimate', 'Least square');

zoomin  = @(f) f(1:100, 1:50);
extract = @(f) zoomin(reshape(f, n1, n2));

%%% Draw matrices
color_interval = 0.008 * [-1 1];
figure
subplot(1,3,1);
imagesc(extract(f0));
axis image;
caxis(color_interval);
set(gca, 'XTick', []);
set(gca, 'YTick', []);
xlabel('x_0');
subplot(1,3,2);
m = phi0.ML(y);
m(m == 0) = nan;
imagescnan(extract(m), 'NanColor', [1 1 1]);
axis image;
caxis(color_interval);
set(gca, 'XTick', []);
set(gca, 'YTick', []);
xlabel('x_{ML}(y)');
subplot(1,3,3);
imagesc(extract(f_bfgs_opt));
axis image;
caxis(color_interval);
set(gca, 'XTick', []);
set(gca, 'YTick', []);
xlabel('x(y, \lambda)');
linkaxes

deterministic('off', state);