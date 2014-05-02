function varargout = ist(y, lambda, ...
                         phi, amplitude, ...
                         stop_func, ...
                         varargin)

% ist - iterative soft-thresholding algorithm
%
%   solve Argmin_x 1/2 || y - \Phi x ||^2 + ||x||_(1,lambda)
%
%   x                = ist(y, lambda, phi, amplitude, stop_func)
%
%   [x, dx_dy_delta] = ist(y, lambda, phi, amplitude, stop_func,
%                          'AppliedJacY', delta)
%
%   [x, dx_dlambda]  = ist(y, lambda, phi, amplitude, stop_func,
%                          'JacTheta')
%
%   y is the input image.
%   lambda is the threshold.
%   phi is the observation matrix.
%   amplitude is the function such that ||x||_(1,lambda) = sum amplitude(x_i,lambda).
%   stop_func is an anonymous function of 3 parameters (t, e, u):
%      t is the iteration index,
%      e is the energy value,
%      u is the relative distance between two iterates,
%      st, the iterations stops as soon as stop_func returns true.
%   delta is the direction is which to apply the jacobian wrt y.
%
%   x is the solution.
%   dx_dy_delta is the jacobian wrt to y in the direction delta
%   dx_dlambda is the jacobian wrt to lambda.
%
%   Copyright (c) 2014 Charles Deledalle

state = deterministic('on');

N = length(phi.AS(y));

% Energy
energy_func = @(x) ...
    norm(y - phi.A(x))^2 + ...
    lambda * sum(amplitude.def(x, lambda));

% Gradient
gradF.def  = @(x, y)     phi.AS(y - phi.A(x));
gradF.D{1} = @(x, y, dx) -phi.AS(phi.A(dx));
gradF.D{2} = @(x, y, dy) phi.AS(dy);

gradF.libschitz_constant = phi.ASA_norm;

% Soft-thresholding
proxG.def  = @(x, y, lambda, tau) ...
    iif(amplitude.def(x, lambda) < tau, ...
        zeros(size(x)), ...
        x .* (1 - lambda * tau ./ amplitude.def(x, lambda)));
proxG.D{1} = @(x, y, lambda, tau, dx) ...
    iif(amplitude.def(x, lambda) < tau, ...
        zeros(size(x)), ...
        dx .* (1 - tau ./ amplitude.def(x, lambda)) + ...
        tau .* x .* amplitude.D{1}(x, lambda, dx) ./ amplitude.def(x, lambda).^2);
proxG.D{2} = @(x, y, lambda, tau, dy) ...
    zeros(size(x));
proxG.D{3} = @(x, y, lambda, tau) ...
    iif(amplitude.def(x, lambda) < tau, ...
        zeros(size(x)), ...
        tau * x ./ amplitude.def(x, lambda).^2 .* amplitude.D{2}(a, lambda));

% Run forward-backward
switch nargout
  case 1
    [varargout{1}] = ...
        fb(y, lambda, N, ...
           gradF, proxG, ...
           stop_func, energy_func, ...
           varargin{:});
  case 2
    [varargout{1} varargout{2}] = ...
        fb(y, lambda, N, ...
           gradF, proxG, ...
           stop_func, energy_func, ...
           varargin{:});
end

deterministic('off', state);