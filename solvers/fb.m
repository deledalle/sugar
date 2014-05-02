function varargout = fb(y, theta, N, ...
                        gradF, proxG, ...
                        stop_func, energy_func, ...
                        varargin)

% fb - forward backward algorithm
%
%   solve Argmin_x F(x, y) + G(x, y, theta)
%
%   x                = fb(y, theta, N, gradF, proxG,
%                         stop_func, energy_func)
%
%   [x, dx_dy_delta] = fb(y, theta, N, gradF, proxG,
%                         stop_func, energy_func,
%                         'AppliedJacY', delta)
%
%   [x, dx_dtheta]   = fb(y, theta, N, gradF, proxG,
%                         stop_func, energy_func,
%                         'JacTheta')
%
%   y is the input image.
%   theta is the parameter of G,
%   N is the size of the vector to recover x.
%   gradF is the gradient of F,
%   proxG is the proximal operator of G,
%   stop_func is an anonymous function of 3 parameters (t, e, u):
%      t is the iteration index,
%      e is the energy value,
%      u is the relative distance between two iterates.
%      st, the iterations stops as soon as stop_func returns true.
%   energy_func is an anonymous function evaluating the energy of a solution x.
%
%   x is the recovered vector.
%   dx_dy_delta is the jacobian wrt to y in the direction delta
%   dx_dtheta is the jacobian wrt to theta.
%
%   Copyright (c) 2014 Charles Deledalle

    %% Parameters
    T = length(theta);
    tau = 1.9 / gradF.libschitz_constant;

    %% Gamma
    gamma.def          = @(x) x;
    gamma.D            = @(x, dx) dx;
    gamma.zero         = zeros(N, 1);
    gamma.dzero_dy_z   = zeros(N, 1);
    gamma.dzero_dtheta = zeros(N, T);

    %% Psi
    function varargout = psi(x, y, theta, varargin)

        optargin = size(varargin, 2);
        if optargin == 0
            differentiation_mode = 'None';
        else
            differentiation_mode = varargin{1};
        end
        switch differentiation_mode
          case 'None'
            x = proxG.def(x - tau * gradF.def(x, y), ...
                          y, ...
                          theta, ...
                          tau);
          case 'AppliedJacY'
            dx_dy_delta = varargin{2};
            delta = varargin{3};

            dX_dy_delta = ...
                dx_dy_delta - ...
                tau * (gradF.D{1}(x, y, dx_dy_delta) + ...
                       gradF.D{2}(x, y, delta));
            X = x - tau * gradF.def(x, y);
            dx_dy_delta = ...
                proxG.D{1}(X, ...
                           y, ...
                           theta, ...
                           tau, ...
                           dX_dy_delta) + ...
                proxG.D{2}(X, ...
                           y, ...
                           theta, ...
                           tau, ...
                           delta);
            x = proxG.def(X, ...
                          y, ...
                          theta, ...
                          tau);
            varargout{2} = dx_dy_delta;
          case 'JacTheta'
            dx_dtheta = varargin{2};
            dX_dtheta = ...
                dx_dtheta - tau * gradF.D{1}(x, y, dx_dtheta);
            X = x - tau * gradF.def(x, y);
            dx_dtheta = ...
                proxG.D{1}(X, ...
                           y, ...
                           theta, ...
                           tau, ...
                           dX_dtheta) + ...
                proxG.D{3}(X, ...
                           y, ...
                           theta, ...
                           tau);
            x = proxG.def(X, ...
                          y, ...
                          theta, ...
                          tau);
            varargout{2} = dx_dtheta;
        end
        varargout{1} = x;
    end

    % Run iterative-scheme
    switch nargout
      case 1
        [varargout{1}] = ...
            iterative_scheme(@psi, gamma, ...
                             y, theta, N, ...
                             stop_func, ...
                             energy_func, ...
                             'Low', ...
                             varargin{:});
      case 2
        [varargout{1} varargout{2}] = ...
            iterative_scheme(@psi, gamma, ...
                             y, theta, N, ...
                             stop_func, ...
                             energy_func, ...
                             'Low', ...
                             varargin{:});
    end
end