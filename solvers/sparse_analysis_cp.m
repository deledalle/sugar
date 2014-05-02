function varargout = sparse_analysis_cp(y, lambda, ...
                                        phi0, psi, D, amplitude, ...
                                        stop_func, ...
                                        varargin)

% sparse_analysis_cp - solve a sparse analysis problem with Chambolle Pock
%
%   solve f^* = \Psi x^*
%   where x^* = Argmin_x 1/2 | y - \Phi0 \Psi x |_2^2 + | D^* x |_(1,lambda)
%   using Chambolle-Pock algorithm where:
%       F(x, y)    = | x |_(1,lambda)
%       G(x, y)    = 1/2 | y - \Phi x |_2^2
%       K          = D^*
%
%   x                = sparse_analysis_cp(y, lambda, phi0, psi, D;
%                                         amplitude, stop_func)
%
%   [x, dx_dy_delta] = sparse_analysis_cp(y, lambda, phi0, psi, D;
%                                         amplitude, stop_func,
%                                         'AppliedJacY', delta)
%
%   [x, dx_dlambda]  = sparse_analysis_cp(y, lambda, phi0, psi, D;
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
    PsiS = psi.AS;
    Psi_PseudoInv = psi.A_PseudoInv;
    Phi0 = phi0.A;
    Phi0S = phi0.AS;
    Phi = @(x) Phi0(Psi(x));
    PhiS = @(y) PsiS(Phi0S(y));
    IdPDDS_Inv = D.IdPAAS_Inv;
    Pi = phi0.Pi;

    %% Length
    P = size(y, 1);
    Nx = size(PhiS(y), 1);
    Nu = size(D.AS(PhiS(y)), 1);
    T = length(lambda);

    %% F(x, y) = | x |_1
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

    proxF.def = @(x,  y, lambda, tau)      SoftThresh.def(x, lambda, tau);
    proxF.D{1} = @(x, y, lambda, tau, dx)  SoftThresh.D{1}(x, lambda, tau, dx);
    proxF.D{2} = @(x, y, lambda, tau, dy)  zeros(Nu, 1);
    proxF.D{3} = @(x, y, lambda, tau)      SoftThresh.D{2}(x, lambda, tau);

    proxFS.def = @(x, y, lambda, gamma) ...
        x - gamma * proxF.def(x/gamma, y, lambda, 1/gamma);
    proxFS.D{1} = @(x, y, lambda, gamma, dx) ...
        dx - proxF.D{1}(x/gamma, y, lambda, 1/gamma, dx);
    proxFS.D{2} = @(x, y, lambda, gamma, dy) ...
        - gamma * proxF.D{2}(x/gamma, y, lambda, 1/gamma, dy);
    proxFS.D{3} = @(x, y, lambda, gamma) ...
        - gamma * proxF.D{3}(x/gamma, y, lambda, 1/gamma);

    %% G(x, y) = 1/2 | y - \Phi x |_2^2
    if phi0.normalized_tight_frame
        proxG.def = @(x, y, tau) ...
            x + tau * PhiS(y - Phi(x)) / (1 + tau);
        proxG.D{1} = @(x, y, tau, dx) ...
            dx - tau * PhiS(Phi(dx)) / (1 + tau);
        proxG.D{2} = @(x, y, tau, dy) ...
            tau * PhiS(dy) / (1 + tau);
    else
        proxG.def = @(x, y, tau) ...
            x + tau * PhiS(y - phi0.IdPtauAAS_Inv(Phi(x + tau * PhiS(y)), tau));
        proxG.D{1} = @(x, y, tau, dx) ...
            dx - tau * PhiS(phi0.IdPtauAAS_Inv(Phi(dx), tau));
        proxG.D{2} = @(x, y, tau, dy) ...
            tau * PhiS(dy - phi0.IdPtauAAS_Inv(Phi(tau * PhiS(dy)), tau));
    end

    %% K = D^*
    K.A = D.AS;
    K.AS = D.A;
    K.ASA_norm = D.AAS_norm;

    %% Energy
    energy_func = @(x) ...
        1/2 * norm(y - Phi(x))^2 + ...
        sum(amplitude.energy(D.AS(x), lambda));

    %% Run Champbolle-Pock
    switch nargout
      case 1
        [varargout{1}] = ...
            chambolle_pock(y, lambda, Nx, ...
                           proxFS, proxG, K, ...
                           stop_func, energy_func, ...
                           varargin{:});
      case 2
        [varargout{1} varargout{2}] = ...
            chambolle_pock(y, lambda, Nx, ...
                           proxFS, proxG, K, ...
                           stop_func, energy_func, ...
                           varargin{:});
    end
    for k = 1:nargout
        varargout{k} = psi.A(varargout{k});
    end
end
