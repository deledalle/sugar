function varargout = sparse_analysis_gfb(y, theta, ...
                                         phi0, psi, D, amplitude_list, ...
                                         stop_func, ...
                                         varargin)

% sparse_analysis_gfb - solve a sparse analysis problem with GFB
%
%   Solve f^* = \Psi x^*
%   where x^* = Argmin_x 1/2 | y - \Phi0 \Psi x |_2^2 + sum_q | D^* x |_a(q)
%   using Generalized Forward-Backward algorithm where:
%       F(x,  y, u)    = 1/2 | y - \Phi0 \Psi x |_2^2
%       Gq(x, y, u)    = | u |_a(q)
%       GQ(x, y, u)    = is(DS * x = u)
%
%   x                = sparse_analysis_gfb(y, lambda, phi0, psi, D;
%                                          amplitude_list, stop_func)
%
%   [x, dx_dy_delta] = sparse_analysis_gfb(y, lambda, phi0, psi, D;
%                                          amplitude_list, stop_func,
%                                          'AppliedJacY', delta)
%
%   [x, dx_dlambda]  = sparse_analysis_gfb(y, lambda, phi0, psi, D;
%                                          amplitude_list, stop_func,
%                                          'JacTheta')
%
%   y is the input image.
%   lambda is the threshold.
%   phi0 is the observation matrix.
%   psi is a synthesis dictionary.
%   D is an analysis dictionary.
%   amplitude_list is a list of functions such that ||x||_a(q) = sum amplitude{q}(x_i, lambda).
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

    if ~iscell(amplitude_list)
        tmp = amplitude_list;
        clear amplitude_list;
        amplitude_list{1} = tmp;
    end

    %%% Shortcuts
    Psi = psi.A;
    PsiS = psi.AS;
    Psi_PseudoInv = psi.A_PseudoInv;
    Phi0 = phi0.A;
    Phi0S = phi0.AS;
    Phi = @(x) Phi0(Psi(x));
    PhiS = @(y) PsiS(Phi0S(y));
    if D.normalized_tight_frame
        IdPDDS_Inv = @(x) x / 2;
    else
        IdPDDS_Inv = D.IdPAAS_Inv;
    end
    Pi = phi0.Pi;

    %% Length
    Nx = length(PhiS(y));
    Nu = length(D.AS(PhiS(y)));
    N = Nx + Nu;
    T = length(theta);

    %% F([x u], y) = 1/2 | y - \Phi x |_2^2
    gradF.def = @(xu, y) ...
        [ PhiS(Phi(xu(1:Nx)) - y) ; zeros(Nu, 1) ];
    gradF.D{1} = @(xu, y, dxu) ...
        [ PhiS(Phi(dxu(1:Nx, :))) ; zeros(Nu, size(dxu, 2)) ];
    gradF.D{2} = @(xu, y, dy) ...
        [ -PhiS(dy) ; zeros(Nu, 1) ];

    %% Gq([x u], y) = | u |_1
    for q = 1:length(amplitude_list)
        SoftThresh.def = @(a, theta, tau) ...
            iif(amplitude_list{q}.def(a, theta) < tau, ...
                zeros(Nu, 1), ...
                a .* (1 - tau ./ amplitude_list{q}.def(a, theta)));
        SoftThresh.D{1} = @(a, theta, tau, da) ...
            iif(amplitude_list{q}.def(a, theta) * ones(1, size(da, 2)) < tau, ...
                zeros(Nu, size(da, 2)), ...
                da .* ((1 - tau ./ amplitude_list{q}.def(a, theta)) ...
                       * ones(1, size(da, 2))) + ...
                (a * ones(1, size(da, 2))) .* ...
                (tau .* amplitude_list{q}.D{1}(a, theta, da) ./ ...
                 (amplitude_list{q}.def(a, theta) * ones(1, size(da, 2))).^2));
        SoftThresh.D{2} = @(a, theta, tau) ...
            iif(amplitude_list{q}.def(a, theta) * ones(1, T) < tau, ...
                zeros(Nu, T), ...
                tau * (a ./ amplitude_list{q}.def(a, theta).^2 * ones(1, T)) ...
                .* amplitude_list{q}.D{2}(a, theta));

        proxG_list{q}.def = @(xu, y, theta, tau) ...
            [ xu(1:Nx) ; ...
              SoftThresh.def(xu((Nx+1):end), theta, tau) ];
        proxG_list{q}.D{1} = @(xu, y, theta, tau, dxu) ...
            [ dxu(1:Nx, :) ; ...
              SoftThresh.D{1}(xu((Nx+1):end), theta, tau, dxu((Nx+1):end, :)) ];
        proxG_list{1}.D{2} = @(xu, y, theta, tau, dy) ...
            zeros(N, 1);
        proxG_list{q}.D{3} = @(xu, y, theta, tau) ...
            [ zeros(Nx, T) ; ...
              SoftThresh.D{2}(xu((Nx+1):end), theta, tau) ];
    end

    %% GQ(y, [x u]) = is(DS * x = u)
    function xu = funcProxG(xu, tau)
        for k = 1:size(xu, 2)
            x = xu(1:Nx,k);
            u = xu((Nx+1):end,k);
            x = IdPDDS_Inv(x + D.A(u));
            u = D.AS(x);
            xu(:,k) = [x; u];
        end
    end
    Q = length(amplitude_list) + 1;
    proxG_list{Q}.def = @(xu, y, theta, tau) ...
        funcProxG(xu, tau);
    proxG_list{Q}.D{1} =  @(xu, y, theta, tau, dxu) ...
        funcProxG(dxu, tau);
    proxG_list{Q}.D{2} =  @(xu, y, theta, tau, dy) ...
        zeros(N, 1);
    proxG_list{Q}.D{3} =  @(xu, y, theta, tau, dy) ...
        zeros(N, T);

    %% Lipschitz constant
    state = deterministic('on');
    gradF.libschitz_constant = ...
        compute_operator_norm(@(x) PhiS(Phi(x)), randn(Nx,1));
    deterministic('off', state);

    %% Energy
    function e = energy_func(xu)
        e = 1/2 * norm(y - Phi(xu(1:Nx)))^2;
        for q = 1:length(amplitude_list)
            e = e + sum(amplitude_list{q}.energy(D.AS(xu(1:Nx)), theta));
        end
    end

    %% Run douglas-rachford
    switch nargout
      case 1
        [varargout{1}] = ...
            gfb(y, theta, N, ...
                gradF, proxG_list, ...
                stop_func, @energy_func, ...
                varargin{:});
        varargout{1} = varargout{1}(1:Nx,:);
      case 2
        [varargout{1} varargout{2}] = ...
            gfb(y, theta, N, ...
                gradF, proxG_list, ...
                stop_func, @energy_func, ...
                varargin{:});
        varargout{1} = varargout{1}(1:Nx,:);
        varargout{2} = varargout{2}(1:Nx,:);
    end
    for k = 1:nargout
        varargout{k} = psi.A(varargout{k});
    end
end
