function varargout = chambolle_pock(y, theta, N, ...
                                    proxFS, proxG, K, ...
                                    stop_func, energy_func, ...
                                    varargin)

% chambolle_pock - Chambolle Pock algorithm
%
%   solve Argmin_x F(K * x, y, theta) + G(x, y)
%
%   x                = chambolle_pock(y, theta, N,
%                                     proxFS, proxG, K,
%                                     stop_func, energy_func)
%
%   [x, dx_dy_delta] = chambolle_pock(y, theta, N,
%                                     proxFS, proxG, K,
%                                     stop_func, energy_func,
%                                     'AppliedJacY', delta)
%
%   [x, dx_dtheta]   = chambolle_pock(y, theta, N,
%                                     proxFS, proxG, K,
%                                     stop_func, energy_func,
%                                     'JacTheta')
%
%   y is the input image.
%   theta is the collection of parameters of F.
%   N is the size of the vector to recover x.
%   proxFS is the proximal operator of F^*.
%   proxG is the proximal operator of G.
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

    %% Parameters
    NU = length(K.A(zeros(N, 1)));
    T = length(theta);
    L = K.ASA_norm;
    eta = 0.45/sqrt(L);
    tau = 0.45/sqrt(L);
    omega = 1;

    %% Gamma
    gamma.def  = @(a) a.x;
    gamma.D    = @(a, da) da.x;
    gamma.zero.x = zeros(N, 1);
    gamma.zero.u = zeros(NU, 1);
    gamma.zero.xtilde = zeros(N, 1);
    gamma.dzero_dy_z.x = zeros(N, 1);
    gamma.dzero_dy_z.u = zeros(NU, 1);
    gamma.dzero_dy_z.xtilde = zeros(N, 1);
    gamma.dzero_dtheta.x = zeros(N, T);
    gamma.dzero_dtheta.u = zeros(NU, T);
    gamma.dzero_dtheta.xtilde = zeros(N, T);

    %% Core
    function varargout = psi(a, y, theta, varargin)
        x = a.x;
        u = a.u;
        xtilde = a.xtilde;

        optargin = size(varargin, 2);
        if optargin == 0
            differentiation_mode = 'None';
        else
            differentiation_mode = varargin{1};
        end
        switch differentiation_mode
          case 'None'
            x_old = x;
            U = u + eta * K.A(xtilde);
            X = x - tau * K.AS(u);
            u = proxFS.def(U, y, theta, eta);
            x = proxG.def(X, y, tau);
            xtilde = x + omega * (x - x_old);
          case 'AppliedJacY'
            dx_dy_z = varargin{2}.x;
            du_dy_z = varargin{2}.u;
            dxtilde_dy_z = varargin{2}.xtilde;
            z = varargin{3};

            dx_old_dy_z = dx_dy_z;
            x_old = x;
            dU_dy_z = du_dy_z + eta * K.A(dxtilde_dy_z);
            U = u + eta * K.A(xtilde);
            dX_dy_z = dx_dy_z - tau * K.AS(du_dy_z);
            X = x - tau * K.AS(u);
            du_dy_z = ...
                proxFS.D{1}(U, y, theta, eta, dU_dy_z) + ...
                proxFS.D{2}(U, y, theta, eta, z);
            u = proxFS.def(U, y, theta, eta);
            dx_dy_z = ...
                proxG.D{1}(X, y, tau, dX_dy_z) + ...
                proxG.D{2}(X, y, tau, z);
            x = proxG.def(X, y, tau);
            dxtilde_dy_z = dx_dy_z + omega * (dx_dy_z - dx_old_dy_z);
            xtilde = x + omega * (x - x_old);

            varargout{2}.x = dx_dy_z;
            varargout{2}.u = du_dy_z;
            varargout{2}.xtilde = dxtilde_dy_z;
          case 'JacTheta'
            dx_dtheta = varargin{2}.x;
            du_dtheta = varargin{2}.u;
            dxtilde_dtheta = varargin{2}.xtilde;

            dx_old_dtheta = dx_dtheta;
            x_old = x;
            dU_dtheta = du_dtheta + eta * K.A(dxtilde_dtheta);
            U = u + eta * K.A(xtilde);
            dX_dtheta = dx_dtheta - tau * K.AS(du_dtheta);
            X = x - tau * K.AS(u);
            du_dtheta = ...
                proxFS.D{1}(U, y, theta, eta, dU_dtheta) + ...
                proxFS.D{3}(U, y, theta, eta);
            u = proxFS.def(U, y, theta, eta);
            dx_dtheta = ...
                proxG.D{1}(X, y, tau, dX_dtheta);
            x = proxG.def(X, y, tau);
            dxtilde_dtheta = dx_dtheta + omega * (dx_dtheta - dx_old_dtheta);
            xtilde = x + omega * (x - x_old);

            varargout{2}.x = dx_dtheta;
            varargout{2}.u = du_dtheta;
            varargout{2}.xtilde = dxtilde_dtheta;
        end
        varargout{1}.x = x;
        varargout{1}.u = u;
        varargout{1}.xtilde = xtilde;
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
