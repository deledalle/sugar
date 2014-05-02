function op = op_tv(N1, N2, options)

% op_tv - create a TV (2D discrete gradient) operator
%
%   op = op_tv(N1, N2)
%
%   N1, N2 are the dimension of the image to analyse.
%
%   Copyright (c) 2014 Charles Deledalle

    if ~exist('options', 'var')
        options = struct();
    end
    use_haar = getoptions(options, 'haar', 0);
    if use_haar
        op = op_haar_udwt_analysis(N1, N2, 1, 0);
        return;
    end

    o.bound = 'per';
    o.nbdims = 2;

    A =  @(a) -div(a,o);
    AS = @(x) grad(x,o);

    laplacian = zeros(N1, N2);
    laplacian(1,1) = -4;
    laplacian(1,2) = +1;
    laplacian(2,1) = +1;
    laplacian(1,N2) = +1;
    laplacian(N1,1) = +1;
    f_laplacian = fft2(laplacian);
    IdPAAS_Inv = @(x) real(ifft2(fft2(x) ./ (1 - f_laplacian)));

    % Vectorize
    op.A = vect(A, N1, N2, 2);
    op.AS = vect(AS, N1, N2);
    op.IdPAAS_Inv = vect(IdPAAS_Inv, N1, N2);

    global silent;
    silent = ~isempty(silent) && sum(abs(silent)) > 0;
    if ~silent
        disp(['Test total-variation']);
    end
    op = properties_tests(op, N1*N2*2, 1);

