function varargout = iterative_scheme(psi, gamma, ...
                                      y, theta, N, ...
                                      stop_func, ...
                                      energy_func, ...
                                      abstraction_mode, ...
                                      varargin)

% iterative_scheme - general iterative scheme
%
%   x                = iterative_scheme(psi, gamma,
%                                       y, theta, N,
%                                       stop_func, energy_func,
%                                       abstraction_mode)
%
%   [x, dx_dy_delta] = iterative_scheme(psi, gamma,
%                                       y, theta, N,
%                                       stop_func, energy_func,
%                                       abstraction_mode
%                                       'AppliedJacY', delta)
%
%   [x, dx_dtheta]   = iterative_scheme(psi, gamma,
%                                       y, theta, N,
%                                       stop_func, energy_func,
%                                       abstraction_mode
%                                       'JacTheta')
%
%   psi is a function whose set of interested auxiliary variables is a point fix.
%   gamma is the function mapping auxiliary variables to the variable of interest.
%   y is the input observation.
%   theta is the parameter of psi,
%   N is the size of the vector to recover x.
%   stop_func is an anonymous function of 3 parameters (t, e, u):
%      t is the iteration index,
%      e is the energy value,
%      u is the relative distance between two iterates.
%      st, the iterations stops as soon as stop_func returns true.
%   energy_func is an anonymous function evaluating the energy of a solution x.
%   abstraction_mode = { 'Low' | 'High' } defines the level of abstraction of psi.
%
%   x is the recovered vector.
%   dx_dy_delta is the jacobian wrt to y in the direction delta
%   dx_dlambda is the jacobian wrt to lambda.
%
%   Copyright (c) 2014 Charles Deledalle

optargin = size(varargin, 2);
if optargin == 0
    differentiation_mode = 'None';
else
    differentiation_mode = varargin{1};
end

T = length(theta);

% Update
update_func = @(a, a_old) max((a-a_old).^2./max(a_old.^2, 1e-6));

% Core
global silent
silent = ~isempty(silent) && sum(abs(silent)) > 0;
if ~silent
    fprintf('  Iterations:\n');
    l = 0;
end

energy = zeros(10000, 1);
update = zeros(10000, 1);
switch differentiation_mode
  case 'None'
    a = gamma.zero;
    t = 0;
    while t < 2 || ~stop_func(t, energy, update)
        a_old = a;

        switch abstraction_mode
          case 'High'
            a = psi.def(a, y, theta);
          case 'Low'
            a = psi(a, y, theta);
        end

        t = t + 1;
        energy(t) = energy_func(gamma.def(a));
        update(t) = update_func(gamma.def(a), gamma.def(a_old));

        if ~silent
            l = display_error(t, update, energy, l);
        end
    end
    x = gamma.def(a);
    varargout{1} = x;
  case 'AppliedJacY'
    z = varargin{2};
    da_dy_z = gamma.dzero_dy_z;
    a = gamma.zero;
    t = 0;
    while t < 2 || ~stop_func(t, energy, update)
        a_old = a;

        switch abstraction_mode
          case 'High'
            da_dy_z = psi.D{1}(a, y, theta, da_dy_z) + ...
                      psi.D{2}(a, y, theta, z);
            a = psi.def(a, y, theta);
          case 'Low'
            [a da_dy_z] = psi(a, y, theta, 'AppliedJacY', da_dy_z, z);
        end

        t = t + 1;
        energy(t) = energy_func(gamma.def(a));
        update(t) = update_func(gamma.def(a), gamma.def(a_old));

        if ~silent
            l = display_error(t, update, energy, l);
        end
    end
    dx_dy_z = gamma.D(a, da_dy_z);
    x = gamma.def(a);
    varargout{1} = x;
    varargout{2} = dx_dy_z;
  case 'JacTheta'
    da_dtheta = gamma.dzero_dtheta;
    a = gamma.zero;

    t = 0;
    while t < 2 || ~stop_func(t, energy, update)
        a_old = a;

        switch abstraction_mode
          case 'High'
            da_dtheta = psi.D{1}(a, y, theta, da_dtheta) + ...
                psi.D{3}(a, y, theta);
            a = psi.def(a, y, theta);
          case 'Low'
            [a da_dtheta] = psi(a, y, theta, 'JacTheta', da_dtheta);
        end

        t = t + 1;
        energy(t) = energy_func(gamma.def(a));
        update(t) = update_func(gamma.def(a), gamma.def(a_old));

        if ~silent
            l = display_error(t, update, energy, l);
        end
    end
    dx_dtheta = gamma.D(a, da_dtheta);
    x = gamma.def(a);
    varargout{1} = x;
    varargout{2} = dx_dtheta;
  otherwise
    error('differentiation mode unknown');
end
energy = energy(1:t);
update = update(1:t);
if ~silent
    fprintf('\n');
end

function l = display_error(t, update, energy, l)

    if mod(t, 10) == 0
        if l>0 % because \r does not work on Matlab
            for j=1:l
                fprintf('\b'); % backspace
            end
            if strcmp(getenv('TERM'), 'dumb')
                fprintf('\r');
            end
        end
        l = fprintf('  Up: %.6e  En: %.9e  It: %6d', ...
                    update(t), ...
                    energy(t), ...
                    t);
    end
