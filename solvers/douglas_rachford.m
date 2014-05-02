function varargout = douglas_rachford(y, theta, N, ...
                                      proxF, proxG, ...
                                      stop_func, energy_func, ...
                                      varargin)

% douglas_rachford - Douglas Rachford algorithm
%
%   Solve Argmin_x F(x, y, theta) + G(x, y, theta)
%
%   x                = douglas_rachford(y, theta, N,
%                                       proxF, proxG,
%                                       stop_func, energy_func)
%
%   [x, dx_dy_delta] = douglas_rachford(y, theta, N,
%                                       proxF, proxG,
%                                       stop_func, energy_func,
%                                       'AppliedJacY', delta)
%
%   [x, dx_dtheta]   = douglas_rachford(y, theta, N,
%                                       proxF, proxG,
%                                       stop_func, energy_func,
%                                       'JacTheta')
%
%   y is the input image.
%   theta is the collection of parameters of F and G.
%   N is the size of the vector to recover x.
%   proxF is the proximal operator of F.
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
    T = length(theta);
    mu = 1;
    tau = 1;

    %% Define reflexive proximal operator
    rProxF.def = @(x,y,theta,tau)       2*proxF.def(x,y,theta,tau)-x;
    rProxG.def = @(x,y,theta,tau)       2*proxG.def(x,y,theta,tau)-x;
    rProxF.D{1} = @(x,y,theta,tau,dx)   2*proxF.D{1}(x,y,theta,tau,dx)-dx;
    rProxG.D{1} = @(x,y,theta,tau,dx)   2*proxG.D{1}(x,y,theta,tau,dx)-dx;
    rProxF.D{2} = @(x,y,theta,tau,dy)   2*proxF.D{2}(x,y,theta,tau,dy);
    rProxG.D{2} = @(x,y,theta,tau,dy)   2*proxG.D{2}(x,y,theta,tau,dy);
    rProxF.D{3} = @(x,y,theta,tau)      2*proxF.D{3}(x,y,theta,tau);
    rProxG.D{3} = @(x,y,theta,tau)      2*proxG.D{3}(x,y,theta,tau);

    %% Gamma
    gamma.def  = @(a) a.x;
    gamma.D    = @(a, da) da.x;
    gamma.zero.x = zeros(N, 1);
    gamma.zero.xp = zeros(N, 1);
    gamma.dzero_dy_z.x = zeros(N, 1);
    gamma.dzero_dy_z.xp = zeros(N, 1);
    gamma.dzero_dtheta.x = zeros(N, T);
    gamma.dzero_dtheta.xp = zeros(N, T);

    %% Core
    function varargout = psi(a, y, theta, varargin)
        x = a.x;
        xp = a.xp;

        optargin = size(varargin, 2);
        if optargin == 0
            differentiation_mode = 'None';
        else
            differentiation_mode = varargin{1};
        end
        switch differentiation_mode
          case 'None'
            xp = (1-mu/2) * xp + ....
                 mu/2 * rProxG.def(rProxF.def(xp, y, theta, tau), ...
                                   y, theta, tau);
            x = proxF.def(xp, y, theta, tau);
          case 'AppliedJacY'
            dx_dy_z = varargin{2}.x;
            dxp_dy_z = varargin{2}.xp;
            z = varargin{3};

            dX_dy_z = ...
                rProxF.D{1}(xp, y, theta, tau, dxp_dy_z) + ...
                rProxF.D{2}(xp, y, theta, tau, z);
            X = rProxF.def(xp, y, theta, tau);
            dxp_dy_z = ...
                (1-mu/2) * dxp_dy_z + ...
                mu/2 * ...
                (rProxG.D{1}(X, y, theta, tau, dX_dy_z) + ...
                 rProxG.D{2}(X, y, theta, tau, z));
            xp = (1-mu/2) * xp + ...
                 mu/2 * rProxG.def(X, y, theta, tau);
            dx_dy_dz = ...
                proxF.D{1}(xp, y, theta, tau, dxp_dy_z) + ...
                proxF.D{2}(xp, y, theta, tau, z);
            x = proxF.def(xp, y, theta, tau);

            varargout{2}.x = dx_dy_z;
            varargout{2}.xp = dxp_dy_z;
          case 'JacTheta'
            dx_dtheta = varargin{2}.x;
            dxp_dtheta = varargin{2}.xp;

            dX_dtheta = ...
                rProxF.D{1}(xp, y, theta, tau, dxp_dtheta) + ...
                rProxF.D{3}(xp, y, theta, tau);
            X = rProxF.def(xp, y, theta, tau);
            dxp_dtheta = ...
                (1-mu/2) * dxp_dtheta + ...
                mu/2 * ...
                (rProxG.D{1}(X, y, theta, tau, dX_dtheta) + ...
                 rProxG.D{3}(X, y, theta, tau));
            xp = (1-mu/2) * xp + ...
                 mu/2 * rProxG.def(X, y, theta, tau);
            dx_dtheta = ...
                proxF.D{1}(xp, y, theta, tau, dxp_dtheta) + ...
                proxF.D{3}(xp, y, theta, tau);
            x = proxF.def(xp, y, theta, tau);

            varargout{2}.x = dx_dtheta;
            varargout{2}.xp = dxp_dtheta;
        end
        varargout{1}.x = x;
        varargout{1}.xp = xp;
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
