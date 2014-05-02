function varargout = sparse_analysis_dr(y, lambda, ...
                                        phi0, psi, D, amplitude, ...
                                        stop_func, ...
                                        varargin)

% sparse_analysis_dr - solve a sparse analysis problem with Douglas Rachford
%
%   solve f^* = \Psi x^*
%   where x^* = Argmin_x 1/2 | y - \Phi0 \Psi x |_2^2 + | D^* x |_(1,lambda)
%   using Douglas-Rachford algorithm where:
%       F(x, y, lambda, u) = F1(x, y) + F2(u, lambda)
%       F1(x, y)           = 1/2 | y - \Phi x |_2^2
%       F2(u, lambda)      = | u |_(1,lambda)
%       G(x, y, u)         = is(DS * x = u)
%
%   x                = sparse_analysis_dr(y, lambda, phi0, psi, D;
%                                         amplitude, stop_func)
%
%   [x, dx_dy_delta] = sparse_analysis_dr(y, lambda, phi0, psi, D;
%                                         amplitude, stop_func,
%                                         'AppliedJacY', delta)
%
%   [x, dx_dlambda]  = sparse_analysis_dr(y, lambda, phi0, psi, D;
%                                         amplitude, stop_func,
%                                         'JacTheta')
%
%   y is the input image.
%   lambda is the threshold.
%   phi0 is the observation matrix.
%   psi is a synthesis dictionary.
%   D is an analysis dictionary.
%   amplitude is the function such that ||x||_(1,lambda) = sum amplitude(x_i, lambda).
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



    if ~psi.normalized_tight_frame
        error('Not implemented yet: Psi is not a normalized tight frame');
    end

    %%% Shortcuts
    Psi = psi.A;
    PsiS = psi.A_PseudoInv;
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
    T = length(lambda);

    %% F1(x, y) = 1/2 | y - \Phi x |_2^2
    if phi0.normalized_tight_frame
        proxF1.def = @(x, y, tau) ...
            x + tau * PhiS(y - Phi(x)) / (1 + tau);
        proxF1.D{1} = @(x, y, tau, dx) ...
            dx - tau * PhiS(Phi(dx)) / (1 + tau);
        proxF1.D{2} = @(x, y, tau, dy) ...
            tau * PhiS(dy) / (1 + tau);
    else
        proxF1.def = @(x, y, tau) ...
            x + tau * PhiS(y - phi0.IdPtauAAS_Inv(Phi(x + tau * PhiS(y)), tau));
        proxF1.D{1} = @(x, y, tau, dx) ...
            dx - tau * PhiS(phi0.IdPtauAAS_Inv(Phi(dx), tau));
        proxF1.D{2} = @(x, y, tau, dy) ...
            tau * PhiS(dy - phi0.IdPtauAAS_Inv(Phi(tau * PhiS(dy)), tau));
    end

    %% F2(u, lambda) = | u |_1
    SoftThresh.def = @(a, lambda, tau) ...
        iif(amplitude.def(a, lambda) < tau, ...
            zeros(Nu, 1), ...
            a .* (1 - tau ./ amplitude.def(a, lambda)));
    SoftThresh.D{1} = @(a, lambda, tau, da) ...
        iif(amplitude.def(a * ones(1, size(da, 2)), lambda) < tau, ...
            zeros(Nu, size(da, 2)), ...
            da .* ((1 - tau ./ amplitude.def(a, lambda)) ...
                   * ones(1, size(da, 2))) + ...
            (a * ones(1, size(da, 2))) .* ...
            (tau .* amplitude.D{1}(a, lambda, da) ./ ...
             (amplitude.def(a, lambda) * ones(1, size(da, 2))).^2));
    SoftThresh.D{2} = @(a, lambda, tau) ...
        iif(amplitude.def(a * ones(1, T), lambda) < tau, ...
            zeros(Nu, T), ...
            tau * (a ./ amplitude.def(a, lambda).^2 * ones(1, T)) ...
            .* amplitude.D{2}(a, lambda));

    proxF2.def = @(u, y, lambda, tau)      SoftThresh.def(u, lambda, tau);
    proxF2.D{1} = @(u, y, lambda, tau, du) SoftThresh.D{1}(u, lambda, tau, du);
    proxF2.D{2} = @(u, y, lambda, tau, dy) zeros(Nu, 1);
    proxF2.D{3} = @(u, y, lambda, tau)     SoftThresh.D{2}(u, lambda, tau);

    %% F(y, [x u]) = F1(x, y) + F2(u, y)
    proxF.def = @(xu, y, lambda, tau) ...
        [ proxF1.def(xu(1:Nx), y, tau) ; ...
          proxF2.def(xu((Nx+1):end), y, lambda, tau) ];
    proxF.D{1} =  @(xu, y, lambda, tau, dxu) ...
        [ proxF1.D{1}(xu(1:Nx), y, tau, dxu(1:Nx,:)) ; ...
          proxF2.D{1}(xu((Nx+1):end), y, lambda, tau, dxu((Nx+1):end,:)) ];
    proxF.D{2} =  @(xu, y, lambda, tau, dy) ...
        [ proxF1.D{2}(xu(1:Nx), y, tau, dy) ; ...
          proxF2.D{2}(xu((Nx+1):end), y, lambda, tau, dy) ];
    proxF.D{3} =  @(xu, y, lambda, tau) ...
        [ zeros(Nx, T) ; ...
          proxF2.D{3}(xu((Nx+1):end), y, lambda, tau) ];

    %% G(y, [x u]) = is(DS * x = u)
    function xu = funcProxG(xu, tau)
        for k = 1:size(xu, 2)
            x = xu(1:Nx,k);
            u = xu((Nx+1):end,k);
            x = IdPDDS_Inv(x + D.A(u));
            u = D.AS(x);
            xu(:,k) = [x; u];
        end
    end
    proxG.def = @(xu, y, lambda, tau) ...
        funcProxG(xu, tau);
    proxG.D{1} =  @(xu, y, lambda, tau, dxu) ...
        funcProxG(dxu, tau);
    proxG.D{2} =  @(xu, y, lambda, tau, dy) ...
        zeros(N, 1);
    proxG.D{3} =  @(xu, y, lambda, tau) ...
        zeros(N, T);

    %% Energy
    energy_func = @(xu) ...
        1/2 * norm(y - Phi(xu(1:Nx)))^2 + ...
        sum(amplitude.energy(D.AS(xu(1:Nx)), lambda));

    %% Run douglas-rachford
    switch nargout
      case 1
        [varargout{1}] = ...
            douglas_rachford(y, lambda, N, ...
                             proxF, proxG, ...
                             stop_func, energy_func, ...
                             varargin{:});
        varargout{1} = varargout{1}(1:Nx,:);
      case 2
        [varargout{1} varargout{2}] = ...
            douglas_rachford(y, lambda, N, ...
                             proxF, proxG, ...
                             stop_func, energy_func, ...
                             varargin{:});
        varargout{1} = varargout{1}(1:Nx,:);
        varargout{2} = varargout{2}(1:Nx,:);
    end
    for k = 1:nargout
        varargout{k} = psi.A(varargout{k});
    end
end
