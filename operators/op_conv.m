function op = op_conv(N1, N2, h)

% op_conv - create a truncated Gaussian convolution
%
%   op = op_conv(N1, N2, h)
%
%   N1, N2 are the dimension of the image to analyse.
%   h is the width of the Gaussian kernel.
%
%   Copyright (c) 2014 Charles Deledalle

    [X Y]                   = meshgrid(((-N2/2):(N2/2-1))', ((-N1/2):(N1/2-1))');
    epsilon                 = 0.1;
    if h == 0
        hg                  = ones(N1,N2);
    else
        g                   = exp(-(X.^2+Y.^2)/h^2);
        g                   = g / sum(g(:));
        hg                  = fft2(fftshift(g));
        hg(hg < epsilon)    = 0;
    end
    proj_g = (hg > 0);
    pseudoinverse_gS = (hg > 0)./max(hg, epsilon);
    pseudoinverse_ggS = (hg > 0)./max(hg.^2, epsilon.^2);
    hg2 = hg.^2;

    A   = @(x) real(ifft2(fft2(x).*hg));
    AS  = @(x) A(x);
    A_Proj  = @(x) real(ifft2(fft2(x).*proj_g));
    AS_PseudoInv  = @(x)  ...
        real(ifft2(fft2(x).*pseudoinverse_gS));
    AAS_PseudoInv  = @(x)  ...
        real(ifft2(fft2(x).*pseudoinverse_ggS));
    IdPtauAAS_Inv = @(x, tau) ...
        real(ifft2(fft2(x)./(1+tau*hg2)));

    % Projection
    Pi                  = @(x) A_Proj(x);
    ML                  = @(y) AS_PseudoInv(y);

    % Vectorize
    op.A = vect(A, N1, N2);
    op.AS = vect(AS, N1, N2);
    op.AS_PseudoInv = vect(AS_PseudoInv, N1, N2);
    op.AAS_PseudoInv = vect(AAS_PseudoInv, N1, N2);
    op.Pi = vect(Pi, N1, N2);
    op.ML = vect(ML, N1, N2);

    function res = IdPtauAAS_Inv_wrap(x, tau)
        f = vect(@(x) IdPtauAAS_Inv(x, tau), N1, N2);
        res = f(x);
    end

    op.IdPtauAAS_Inv = @(x, tau) IdPtauAAS_Inv_wrap(x, tau);

    global silent;
    silent = ~isempty(silent) && sum(abs(silent)) > 0;
    if ~silent
        disp(['Test blur 2D']);
    end
    op = properties_tests(op, N1 * N2, 1);

end