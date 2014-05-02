function varargout = estimate_risk_fdmc(y, theta_list, ...
                                        param, phi0, sig, ...
                                        solver, risk, ...
                                        varargin)

% estimate_risk_fdmc - estimate the risk with finite differences
%                      monte carlo simulation
%
%   [sure, x]        = estimate_risk_fdmc(y, theta,
%                                         param, phi0, sig,
%                                         solver, risk)
%
%   [sure, sugar, x] = estimate_risk_fdmc(y, theta,
%                                         param, phi0, sig,
%                                         solver, risk,
%                                         'sugar')
%
%   y is an observed vector.
%   theta is a parameter or a list of parameters.
%   param is the parameter manager.
%   phi0 is the observation operator.
%   sig is the standard deviation of the noise.
%   solver is the function giving the solution from y and theta.
%   risk is the risk manager.
%
%   sure is the risk or a list of risk estimates associated to each theta.
%   x is the solution or a list of solutions associated to each theta.
%   sugar is the gradient or the list of gradient of the risk estimates.
%
%   Copyright (c) 2014 Charles Deledalle

switch nargout
  case 2
    compute_sugar = 0;
  case 3
    compute_sugar = 1;
  otherwise
    error('Unexpected number of output arguments');
end
if length(varargin) > 0 && strcmp(varargin{1}, 'sugar')
    compute_sugar = 1;
end
P = length(y);
f_ml = phi0.ML(y);
T = size(theta_list, 1);
epsilon = 2*P^(-0.3)*sig;
delta = risk.delta;

switch risk.type
  case 'prediction'
    A   = @(y) y;
    AS  = @(x) x;
    ASA = @(y) y;
    ASA_trace = P;
  case 'projection'
    A   = @(y) phi0.AS(phi0.AAS_PseudoInv(y));
    AS  = @(x) phi0.AAS_PseudoInv(phi0.A(x));
    ASA = @(y) phi0.AAS_PseudoInv(y);
    ASA_trace = phi0.AAS_PseudoInv_trace;
  case 'estimation'
    A   = @(y) phi0.ASA_Inv(phi0.AS(y));
    AS  = @(x) phi0.A(phi0.ASA_Inv(x));
    ASA = @(y) AS(A(y));
    ASA_trace = phi0.ASA_trace;
end

global silent
silent = ~isempty(silent) && sum(abs(silent)) > 0;

sure_list = zeros(1, size(theta_list, 2));
sugar_list = zeros(size(theta_list));
for k = 1:size(theta_list, 2)
    theta = theta_list(:, k);

    if ~silent
        fprintf('\nParam: ');
        param.show(theta);
        fprintf('\n');
    end
    if ~param.ok(theta);
        f = f_ml;
        sure = inf;
        sugar = nan;
    else
        if ~compute_sugar
            f = ...
                solver(y, theta);
            fpd = ...
                solver(y + epsilon * delta, theta);
        else
            [f, df_dtheta] = ...
                solver(y, theta);
            [fpd, dfpd_dtheta] = ...
                solver(y + epsilon * delta, theta);
        end
        % SURE
        sure = ...
            norm(A(y - phi0.A(f)))^2 + ...
            - sig^2 * ASA_trace + ...
            2 * sig^2 * ASA(delta)' * phi0.A(fpd - f) / epsilon;

        % SUGAR
        if compute_sugar
            sugar = ...
                2 * df_dtheta' * phi0.AS(ASA(phi0.A(f) - y)) + ...
                2 * sig^2  / epsilon * ...
                phi0.A(dfpd_dtheta - df_dtheta)' * ASA(delta);
        end

        % Normalization
        sure = sure / (sig^2 * ASA_trace);
        if compute_sugar
            sugar = sugar / (sig^2 * ASA_trace);
        end
    end
    if ~silent
        fprintf('  SURE:  %.6e\n', sure);
        if compute_sugar
            fprintf('  SUGAR:'); fprintf(' %.6e', sugar); fprintf(['\n']);
        end
    end
    sure_list(k) = sure;
    if compute_sugar
        sugar_list(:,k) = sugar;
    end
    f_list{k} = f;
end
if ~compute_sugar
    varargout{1} = sure_list;
    varargout{2} = f_list;
else
    varargout{1} = sure_list;
    varargout{2} = sugar_list;
    if nargout == 3
        varargout{3} = f_list;
    end
end
