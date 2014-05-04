% gen_fig51a - produces Figure 5.1 (a) of the SUGAR paper
%
%   Copyright (c) 2014 Charles Deledalle

clear all
close all

addpathrec('.');
state = deterministic('on');

%%% Load problem settings
setting_matrix_completion;

%%% Define range of lambdas
lmin = 0.0018;
lmax = 0.0479;
lnb = 26;
lambda_list = linspace(lmin, lmax, lnb);

%%% Estimate SURE-MC with Exhaustive search
fprintf('\nStart exhaustive search (%d points, may take some time)\n', lnb);
[pred_sure, f] = estimate_risk_mc(y, lambda_list, ...
                                  param, phi0, sig, ...
                                  solver_for_mc, risk);
[~, k_opt] = min(pred_sure);
lambda_opt = lambda_list(k_opt);

%%% Compute Squared Errors
for k = 1:length(lambda_list)
    se(k) = norm(f{k} - f0)^2 / P;
    pred_se(k) = norm(phi0.A(f{k} - f0))^2 / (sig^2 * P);
end

%%% Draw Risk curves
ratio = 16/9;
scaling = ...
    (max(lambda_list) - min(lambda_list)) / ...
    (max(pred_sure) - min(pred_sure)) / ...
    ratio;

figure,
a = abs(pred_sure(floor(end/5)) - pred_sure(floor(end/2))) ...
    / abs(se(floor(end/5)) - se(floor(end/2)));
t = a * se(k_opt) - pred_sure(k_opt);
plot(lambda_list / scaling, a*se - t, ...
     'r-.');
hold on
t = mean(pred_se - pred_sure);
plot(lambda_list / scaling, pred_se - t, ...
     'k--', 'LineWidth', 1);
plot(lambda_list / scaling, pred_sure, ...
     'k-', 'LineWidth', 1);
plot(lambda_list((2-mod(k_opt,2):2:(end-2))) / scaling, pred_sure((2-mod(k_opt,2)):2:(end-2)), ...
     'xb', 'LineWidth', 2, 'MarkerSize', 10);
plot(lambda_list(k_opt) / scaling, pred_sure(k_opt), ...
     'xr', 'LineWidth', 2, 'MarkerSize', 10);
axis image
xlim([min(lambda_list) max(lambda_list)] / scaling);
m = (max(pred_sure) - min(pred_sure)) / 10;
ylim([min(pred_sure)-m max(pred_sure)+m]);
step_tick = (lmax-lmin)/5;
lambda_tick = (lmin+step_tick):step_tick:(lmax-step_tick);
set(gca, 'XTick', lambda_tick / scaling);
set(gca, 'XTickLabel', ...
         mycellfun(@(f)  sprintf('%.3f', f), ...
                   num2cell(lambda_tick)));
xlabel('Regularization parameter \lambda');
ylabel('Quadratic cost');
legend('Risk', 'Prediction risk', 'SURE_{MC}', ...
       'Exh. search iterates', ...
       'Exh. search final');

%%% Draw spectrums
figure,
semilogx(svd(reshape(f0, n1, n2)), 'r', 'LineWidth', 2);
hold on
plot(svd(reshape(f{k_opt}, n1, n2)), 'b', 'LineWidth', 2);
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
imagesc(extract(f{k_opt}));
axis image;
caxis(color_interval);
set(gca, 'XTick', []);
set(gca, 'YTick', []);
xlabel('x(y, \lambda)');
linkaxes

deterministic('off', state);
