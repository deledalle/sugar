% table1 - produces the table 1 of the SUGAR paper
%
%   Copyright (c) 2014 Charles Deledalle

clear all
close all

addpathrec('.');

for k = 1:3
    switch k
      case 1
        exp{k}{1}.img = 'mandrill';
      case 2
        exp{k}{1}.img = 'house';
      case 3
        exp{k}{1}.img = 'cameraman';
    end
    exp{k}{1}.J = 1;
    exp{k}{1}.mode = 'global';
    exp{k}{2} = exp{k}{1};
    exp{k}{2}.J = 2;
    exp{k}{3} = exp{k}{1};
    exp{k}{3}.J = 3;
    exp{k}{4} = exp{k}{2};
    exp{k}{4}.mode = 'all';
    exp{k}{5} = exp{k}{3};
    exp{k}{5}.mode = 'all';
end

global silent
silent = true;
for k = 1:length(exp)
    for l = 1:length(exp{k})
        state = deterministic('on');

        %%% Set experiment
        f0 = double(imread(sprintf('%s.png', exp{k}{l}.img)));
        if strcmp(exp{k}{l}.img, 'mandrill')
            f0 = f0(50 + (1:256), 50 + (1:256));
        end
        J = exp{k}{l}.J;

        %%% Load problem settings
        switch exp{k}{l}.mode
          case 'global'
            setting_multiscale_globalparam;
          case 'all'
            setting_multiscale;
        end

        %%% Optimize SURE-FDMC with quasi-Newton
        lambda_ini = param.init;
        objective = @(lambda) estimate_risk_fdmc(y, lambda, param, phi0, sig, ...
                                                 solver_for_fdmc, risk, 'sugar');
        objective_sugarfree = @(lambda) estimate_risk_fdmc(y, lambda, param, phi0, sig, ...
                                                          solver_for_fdmc, risk);
        tic;
        [lambda_bfgs_opt, ~, ~, N_rec, ...
         lambda_bfgs_rec, pred_sure_bfgs_rec, pred_sugar_bfgs_rec] = ...
            perform_bfgs(objective, lambda_ini);
        time_bfgs = toc;

        %%% Show stats
        alpharange = [0.75 1 1.25];
        [asure, af] = objective_sugarfree(repmat(lambda_bfgs_opt, [1 3]) * diag(alpharange));
        fprintf('%d \t (', N_rec);
        param.show(lambda_bfgs_opt)
        fprintf(')');
        fprintf('\t %.2e/%.2f', asure(1), psnr(af{1}, f0));
        fprintf('\t %.2e/%.2f', asure(2), psnr(af{2}, f0));
        fprintf('\t %.2e/%.2f', asure(3), psnr(af{3}, f0));
        fprintf('\n');

        deterministic('off', state);
    end
end