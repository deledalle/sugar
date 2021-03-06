function varargout = gfb(y, theta, N, ...
                         gradF, proxG_list, ...
                         stop_func, energy_func, ...
                         varargin)

% gfb - Generalized forward-backward algorithm (Raguet et al. 2013)
%
%   solve Argmin_x F(x, y) + \sim_i G_i(x, y, theta)
%
%   x                = gfb(y, theta, N,
%                          gradF, proxG_list,
%                          stop_func, energy_func)
%
%   [x, dx_dy_delta] = gfb(y, theta, N,
%                          gradF, proxG_list,
%                          stop_func, energy_func,
%                          'AppliedJacY', delta)
%
%   [x, dx_dtheta]   = gfb(y, theta, N,
%                          gradF, proxG_list,
%                          stop_func, energy_func,
%                          'JacTheta')
%
%   y is the input image.
%   theta is the collection of parameters of the G_i.
%   N is the size of the vector to recover x.
%   gradF is the gradient of F.
%   proxG_list is a cell array of proximal operators of G_i.
%   stop_func is an anonymous function of 3 parameters (t, e, u):
%      t is the iteration index,
%      e is the energy value,
%      u is the relative distance between two iterates.
%      st, the iterations stops as soon as stop_func returns true.
%   energy_func is an anonymous function evaluating the energy at x.
%   delta is the direction is which to apply the jacobian wrt y.
%
%   x is the solution.
%   dx_dy_delta is the jacobian wrt to y in the direction delta
%   dx_dtheta is the jacobian wrt to theta.
%
%   Copyright (c) 2014 Charles Deledalle

    if ~iscell(proxG_list)
        tmp = proxG_list;
        clear proxG_list;
        proxG_list{1} = tmp;
    end

    %% Parameters
    T = length(theta);
    tau = 1.9 / gradF.libschitz_constant;
    lambda = 1;
    n = length(proxG_list);
    omega = 1/n;

    %% Gamma
    gamma.def  = @(a) a.x;
    gamma.D    = @(a, da) da.x;
    dzero_dtheta = zeros(N, T);
    for i = 1:n
        zero_list{i} = zeros(N, 1);
        dzero_dtheta_list{i} = zeros(N, T);
    end
    gamma.zero.x = zeros(N, 1);
    gamma.zero.z_list = zero_list;
    gamma.dzero_dy_z.x = zeros(N, 1);
    gamma.dzero_dy_z.z_list = zero_list;
    gamma.dzero_dtheta.x = dzero_dtheta;
    gamma.dzero_dtheta.z_list = dzero_dtheta_list;

    %% Core

    function varargout = psi(a, y, theta, varargin)
        x = a.x;
        z_list = a.z_list;

        optargin = size(varargin, 2);
        if optargin == 0
            differentiation_mode = 'None';
        else
            differentiation_mode = varargin{1};
        end
        switch differentiation_mode
          case 'None'
            gNablaF = tau * gradF.def(x, y);
            for i = 1:n
                X = 2 * x - z_list{i} - gNablaF;
                z_list{i} = z_list{i} + ...
                    lambda * (...
                        proxG_list{i}.def(X, ...
                                          y, ...
                                          theta, ...
                                          tau / omega) ...
                        - x);
            end
            x = zeros(N, 1);
            for i = 1:n
                x = x + omega * z_list{i};
            end
          case 'AppliedJacY'
            dx_dy_z = varargin{2}.x;
            dz_dy_z_list = varargin{2}.z_list;
            z = varargin{3};

            dgNablaF_dy_z = ...
                tau * (gradF.D{1}(x, y, dx_dy_z) + ...
                       gradF.D{2}(x, y, z));
            gNablaF = ...
                tau * gradF.def(x, y);
            for i = 1:n
                dX_dy_z = ...
                    2 * dx_dy_z - dz_dy_z_list{i} ...
                    - dgNablaF_dy_z;
                X = 2 * x - z_list{i} - gNablaF;
                dz_dy_z_list{i} = ...
                    dz_dy_z_list{i} + ...
                    lambda * (...
                        proxG_list{i}.D{1}(X, ...
                                           y, ...
                                           theta, ...
                                           tau / omega, ...
                                           dX_dy_z) + ...
                        proxG_list{i}.D{2}(X, ...
                                           y, ...
                                           theta, ...
                                           tau / omega, ...
                                           z) ...
                        - dx_dy_z);
                z_list{i} = ...
                    z_list{i} + ...
                    lambda * (...
                        proxG_list{i}.def(X, ...
                                          y, ...
                                          theta, ...
                                          tau / omega) ...
                        - x);
            end
            dx_dy_z = zeros(N, 1);
            x = zeros(N, 1);
            for i = 1:n
                dx_dy_z = dx_dy_z + omega * dz_dy_z_list{i};
                x = x + omega * z_list{i};
            end
            varargout{2}.x = dx_dy_z;
            varargout{2}.z_list = dz_dy_z_list;
          case 'JacTheta'
            dx_dtheta = varargin{2}.x;
            dz_dtheta_list = varargin{2}.z_list;

            dgNablaF_dtheta = ...
                tau * gradF.D{1}(x, y, dx_dtheta);
            gNablaF = ...
                tau * gradF.def(x, y);
            for i = 1:n
                dX_dtheta = ...
                    2 * dx_dtheta - dz_dtheta_list{i} ...
                    - dgNablaF_dtheta;
                X = 2 * x - z_list{i} - gNablaF;
                dz_dtheta_list{i} = ...
                    dz_dtheta_list{i} + ...
                    lambda * (...
                        proxG_list{i}.D{1}(X, ...
                                           y, ...
                                           theta, ...
                                           tau / omega, ...
                                           dX_dtheta) + ...
                        proxG_list{i}.D{3}(X, ...
                                           y, ...
                                           theta, ...
                                           tau / omega) ...
                        - dx_dtheta);
                z_list{i} = ...
                    z_list{i} + ...
                    lambda * (...
                        proxG_list{i}.def(X, ...
                                          y, ...
                                          theta, ...
                                          tau / omega) ...
                        - x);
            end
            dx_dtheta = zeros(N, T);
            x = zeros(N, 1);
            for i = 1:n
                dx_dtheta = dx_dtheta + omega * dz_dtheta_list{i};
                x = x + omega * z_list{i};
            end
            varargout{2}.x = dx_dtheta;
            varargout{2}.z_list = dz_dtheta_list;
        end
        varargout{1}.x = x;
        varargout{1}.z_list = z_list;
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
