function varargout = estimate_risk_mc(y, theta_list, ...
                                      param, phi0, sig, ...
                                      solver, risk)

% estimate_risk_mc - estimate the risk with explicit
%                    differentiation and monte carlo simulation
%
%   [sure, x]        = estimate_risk_mc(y, theta,
%                                       param, phi0, sig,
%                                       solver, risk)
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
%
%   Copyright (c) 2014 Charles Deledalle

switch nargout
  case 2
  otherwise
    error('Unexpected number of output arguments');
end
P = length(y);
f_ml = phi0.ML(y);
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
  otherwise
    error('Unexpected risk_type');
end

global silent
silent = ~isempty(silent) && sum(abs(silent)) > 0;

sure_list = zeros(1, size(theta_list, 2));
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
    else
        [f, df_dy_delta] = solver(y, theta, delta);
        % SURE
        sure = ...
            norm(A(y - phi0.A(f)))^2 + ...
            - sig^2 * ASA_trace + ...
            2 * sig^2 * ASA(delta)' * phi0.A(df_dy_delta);

        % Normalization
        sure = sure / (sig^2 * ASA_trace);
    end
    if ~silent
        fprintf('  SURE:  %.6e\n', sure);
    end
    sure_list(k) = sure;
    if size(theta_list, 2) == 1
        f_list = f;
    else
        f_list{k} = f;
    end
end
varargout{1} = sure_list;
varargout{2} = f_list;
