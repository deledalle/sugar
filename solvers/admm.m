function varargout = admm(y, theta, N, ...
                          proxF, proxG, ...
                          stop_func, energy_func, ...
                          varargin)

% ADMM - ADMM Algorithm
%
%   solve Argmin_x F(x, y, theta) + G(x, y)
%
%   x                = admm(y, theta, N,
%                                     proxFS, proxG, K,
%                                     stop_func, energy_func)
%
%   [x, dx_dy_delta] = admm(y, theta, N,
%                                     proxFS, proxG, K,
%                                     stop_func, energy_func,
%                                     'AppliedJacY', delta)
%
%   [x, dx_dtheta]   = admm(y, theta, N,
%                                     proxFS, proxG, K,
%                                     stop_func, energy_func,
%                                     'JacTheta')
%
%   y is the input image.
%   theta is the collection of parameters of F.
%   N is the size of the vector to recover x.
%   proxF is the proximal operator of F.
%   proxG is the proximal operator of G.
%   stop_func is an anonymous function of 3 parameters (t, e, u):
%      t is the iteration index,
%      e is the energy value,
%      u is the relative distance between two iterates.
%      st, the iterations stops as soon as stop_func returns true.
%   energy_func is an anonymous function evaluating the energy at x.
%   delta is the direction in which to apply the jacobian wrt y.
%
%   x is the solution.
%   dx_dy_delta is the jacobian wrt to y in the direction delta
%   dx_dtheta is the jacobian wrt to theta.
%
%   Copyright (c) 2019 Charles Deledalle


    %% Parameters
    T = length(theta);
    tau = 1;

    %% Gamma
    gamma.def  = @(a) a.x;
    gamma.D    = @(a, da) da.x;
    gamma.zero.x = zeros(N, 1);
    gamma.zero.u = zeros(N, 1);
    gamma.zero.d = zeros(N, 1);
    gamma.dzero_dy_z.x = zeros(N, 1);
    gamma.dzero_dy_z.u = zeros(N, 1);
    gamma.dzero_dy_z.d = zeros(N, 1);
    gamma.dzero_dtheta.x = zeros(N, T);
    gamma.dzero_dtheta.u = zeros(N, T);
    gamma.dzero_dtheta.d = zeros(N, T);

    %% Core
    function varargout = psi(a, y, theta, varargin)
        x = a.x;
        u = a.u;
        d = a.d;

        optargin = size(varargin, 2);
        if optargin == 0
            differentiation_mode = 'None';
        else
            differentiation_mode = varargin{1};
        end
        switch differentiation_mode
            case 'None'
                x = proxF.def(u + d, y, theta, tau);
                u = proxG.def(x - d, y, tau);
                d = d - x + u;
            case 'AppliedJacY'
                dx_dy_z = varargin{2}.x;
                du_dy_z = varargin{2}.u;
                dd_dy_z = varargin{2}.d;
                z = varargin{3};

                dx_dy_z = ...
                    proxF.D{1}(u + d, y, theta, tau, du_dy_z + dd_dy_z) + ...
                    proxF.D{2}(u + d, y, theta, tau, z);
                x = proxF.def(u + d, y, theta, tau);
                du_dy_z = ...
                    proxG.D{1}(x - d, y, tau, dx_dy_z - dd_dy_z) + ...
                    proxG.D{2}(x - d, y, tau, z);
                u = proxG.def(x - d, y, tau);
                dd_dy_z = dd_dy_z - dx_dy_z + du_dy_z;
                d = d - x + u;

                varargout{2}.x = dx_dy_z;
                varargout{2}.u = du_dy_z;
                varargout{2}.d = dd_dy_z;
            case 'JacTheta'
                dx_dtheta = varargin{2}.x;
                du_dtheta = varargin{2}.u;
                dd_dtheta = varargin{2}.d;

                dx_dtheta = ...
                    proxF.D{1}(u + d, y, theta, tau, du_dtheta + dd_dtheta) + ...
                    proxF.D{3}(u + d, y, theta, tau);
                x = proxF.def(u + d, y, theta, tau);
                du_dtheta = ...
                    proxG.D{1}(x - d, y, tau, dx_dtheta - dd_dtheta);
                u = proxG.def(x - d, y, tau);
                dd_dtheta = dd_dtheta - dx_dtheta + du_dtheta;
                d = d - x + u;

                varargout{2}.x = dx_dtheta;
                varargout{2}.u = du_dtheta;
                varargout{2}.d = dd_dtheta;
        end
        varargout{1}.x = x;
        varargout{1}.u = u;
        varargout{1}.d = d;
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
        case 3
            [varargout{1} varargout{2} varargout{3}] = ...
                iterative_scheme(@psi, gamma, ...
                                 y, theta, N, ...
                                 stop_func, ...
                                 energy_func, ...
                                 'Low', ...
                                 varargin{:});
        case 4
            [varargout{1} varargout{2} varargout{3} varargout{4}] = ...
                iterative_scheme(@psi, gamma, ...
                                 y, theta, N, ...
                                 stop_func, ...
                                 energy_func, ...
                                 'Low', ...
                                 varargin{:});
    end

 end
