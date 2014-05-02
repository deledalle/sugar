function varargout = nuclearnorm_fb(y, lambda, ...
                                    phi0, n1, n2, ...
                                    stop_func, ...
                                    varargin)

% nuclearnorm_fb - forward backward algorithm for nuclear norm regularization
%
%   solve f^* = \Psi x^*
%   where x^* = Argmin_x 1/2 | y - \Phi0 x |_2^2 + \lambda | M(x) |_*
%   using Forward-Backward algorithm where:
%       F(x, y)         = 1/2 | y - \Phi0 x |_2^2
%       G(x, \lambda)   = \lambda | M(x) |_*
%
%   x                = nuclearnorm_fb(y, lambda, phi0, n1, n2, stop_func)
%
%   [x, dx_dy_delta] = nuclearnorm_fb(y, lambda, phi0, n1, n2, stop_func,
%                                     'AppliedJacY', delta)
%
%   [x, dx_dtheta]   = nuclearnorm_fb(y, lambda, phi0, n1, n2, stop_func
%                                     'JacTheta')
%
%   y is the observed vector.
%   lambda is the threshold.
%   phi0 is the observation operator.
%   n1, n2 are the dimensions of the matrix de recover.
%   stop_func is an anonymous function of 3 parameters (t, e, u):
%      t is the iteration index,
%      e is the energy value,
%      u is the relative distance between two iterates,
%      st, the iterations stops as soon as stop_func returns true.
%   delta is the direction is which to apply the jacobian wrt y.
%
%   x is the recovered matrix.
%   dx_dy_delta is the jacobian wrt to y in the direction delta
%   dx_dlambda is the jacobian wrt to lambda.
%
%   Copyright (c) 2014 Charles Deledalle


    %%% Shortcuts
    Phi  = phi0.A;
    PhiS = phi0.AS;
    Pi   = phi0.Pi;

    %%% Length
    N = size(PhiS(y), 1);
    n = min(n1, n2);
    m = max(n1, n2);

    %% F(x, y) = 1/2 | y - \Phi x |_2^2
    gradF.def = @(x, y) ...
        PhiS(Phi(x) - y);
    gradF.D{1} = @(x, y, dx) ...
        PhiS(Phi(dx));
    gradF.D{2} = @(x, y, dy) ...
        -PhiS(dy);

    %% G(x, \lambda) = \lambda | M(x) |_1
    SoftThresh.def = @(a, lambda, tau) ...
        iif(abs(a) < tau * lambda, ...
            zeros(n, 1), ...
            a - tau * lambda * sign(a));
    SoftThresh.D{1} = @(a, lambda, tau, da) ...
        iif(abs(a * ones(1, size(da, 2))) < tau * lambda, ...
            zeros(length(a), size(da, 2)), ...
            da);
    SoftThresh.D{2} = @(a, lambda, tau) ...
        iif(abs(a) < tau * lambda, ...
            zeros(n, 1), ...
            - tau * sign(a));

    function res = ProxG(u, lambda, tau)
        [V, dLa, U] = svd(reshape(u, n1, n2));
        La = diag(dLa);
        STLa = SoftThresh.def(La, lambda, tau);
        dSTLa = zeros(n1, n2);
        dSTLa(1:n, 1:n) = diag(STLa);
        res = V * dSTLa * U';
        res = res(:);
    end

    [Y, X] = meshgrid(1:n2, 1:n1);

    function res = dProxGdu_z(u, lambda, tau, du);
        [V, dLa, U] = svd(reshape(u, n1, n2));

        La = diag(dLa);
        STLa = SoftThresh.def(La, lambda, tau);

        La0 = zeros(n1, 1);
        La1 = zeros(n1, n2);
        La2 = zeros(n1, n2);
        La0(1:n) = La;
        La1(1:n, :) = La * ones(1, n2);
        La2(:, 1:n) = ones(n1, 1) * La';

        STLa1 = zeros(n1, n2);
        STLa2 = zeros(n1, n2);
        STLa1(1:n, :) = STLa * ones(1, n2);
        STLa2(:, 1:n) = ones(n1, 1) * STLa';

        bard = V' * reshape(du, n1, n2) * U;

        H_z = zeros(n1, n2);
        H_z(1:n, 1:n) = diag(SoftThresh.D{1}(La, lambda, tau, ...
                                             diag(bard(1:n, 1:n))));

        zS = bard / 2;
        zA = bard / 2;
        zS(1:n, 1:n) = (bard(1:n, 1:n) + bard(1:n, 1:n)') / 2;
        zA(1:n, 1:n) = (bard(1:n, 1:n) - bard(1:n, 1:n)') / 2;

        GammaS_z = ...
            iif(La1 == La2, ...
                iif(X <= n, ...
                    SoftThresh.D{1}(La0, lambda, tau, zS), ...
                    zeros(n1, n2)), ...
                (STLa1 - STLa2) ./ (La1 - La2) .* zS);
        GammaA_z = ...
            iif(La1 == -La2, ...
                iif(X <= n, ...
                    SoftThresh.D{1}(La0, lambda, tau, zA), ...
                    zeros(n1, n2)), ...
                (STLa1 + STLa2) ./ (La1 + La2) .* zA);
        GammaS_z(X == Y) = 0;
        GammaA_z(X == Y) = 0;

        res = V * (H_z + GammaS_z + GammaA_z) * U';
        res = res(:);
    end

    function res = dProxG_dlambda(u, lambda, tau)
        n = min(n1, n2);
        [V, dLa, U] = svd(reshape(u, n1, n2));
        La = diag(dLa);
        STLa = SoftThresh.D{2}(La, lambda, tau);
        dSTLa = zeros(n1, n2);
        dSTLa(1:n, 1:n) = diag(STLa);
        res = V * dSTLa * U';
        res = res(:);
    end

    proxG.def = @(x, y, lambda, tau)      ProxG(x, lambda, tau);
    proxG.D{1} = @(x, y, lambda, tau, dx) dProxGdu_z(x, lambda, tau, dx);
    proxG.D{2} = @(x, y, lambda, tau, dy) zeros(size(x));
    proxG.D{3} = @(x, y, lambda, tau, dy) dProxG_dlambda(x, lambda, tau);

    %% Lipschitz constant
    gradF.libschitz_constant = phi0.ASA_norm;

    %% Energy
    energy_func = @(x) ...
        1/2 * norm(y - Phi(x))^2 + ...
        lambda * sum(abs(svd(x)));

    %% Core
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
end
