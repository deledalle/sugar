function op = op_wavelet_udwt_analysis(N1, N2, depth, gamma, mother, child)

% op_wavelet_udwt_analysis - create a Daubechies 4 UDWT analysis operator
%
%   op = op_wavelet_udwt_analysis(N1, N2, depth, gamma, mother, child)
%
%   N1, N2 are the dimension of the image to analyse.
%   depth is a vector of all levels of the scales.
%   gamma defines the weighting of each scale by 4^(d (gamma - 1)).
%   mother and whild define the wavelet shape.
%
%   The obtained dictionary is redundant but is not a complete
%   Daubechies 4 dictionnary. It computes only detail coefficients
%   in vertical and horizontal
%
%   Copyright (c) 2014 Charles Deledalle

L = depth(end);

if ~exist('gamma', 'var')
    gamma = 1;
end

% Create filter-bank
alpha = 4^(gamma);

c = ceil(length(mother) / 2) - 1;
hg_X = zeros(N1, N2);
for i = 1:length(mother)
    hg_X(mod(i - c - 1, N1) + 1, 1) = mother(i);
end
hg_X = hg_X / norm(hg_X);
hg_list{1} = real(ifft2(fft2(hg_X) .* fft2(hg_X')));

hf = zeros(N1, N2);
for i = 1:length(mother)
    hf(mod(i - c - 1, N1) + 1, 1) = child(i);
end
hf = hf / norm(hf);
hf_list{1} = hf;

for k = 2:L
    hg_list{k} = atrous(hg_list{k-1});
    hf_list{k} = atrous(hf_list{k-1});
end

% Approximation coefficients
c = 0;
for k = depth
    % Vertical
    c = c + 1;

    hgp = zeros(N1, N2);
    hgp(1, 1) = 1;
    for i = 1:(k-1)
        hgp = real(ifft2(fft2(hg_list{i}) .* fft2(hgp)));
    end
    hf = real(ifft2(fft2(hf_list{k}) .* fft2(hgp)));
    hf = hf / max(abs(hf(:)));

    f_h{c}.adj = fft2(hf) / 4^k * alpha^k;

    hf = hf([1 end:-1:2], [1 end:-1:2]);
    f_h{c}.def = fft2(hf) / 4^k * alpha^k;
    f_h{c}.k = k;

    % Horizontal
    c = c + 1;

    hf = hf';
    f_h{c}.adj = fft2(hf) / 4^k * alpha^k;

    hf = hf([1 end:-1:2], [1 end:-1:2]);
    f_h{c}.def = fft2(hf) / 4^k * alpha^k;
    f_h{c}.k = k;
end
C = length(f_h);

% For pseudo-inversion
f_i = ones(N1, N2);
for c = 1:C
    k = f_h{c}.k;
    f_i = f_i + abs(f_h{c}.adj).^2;
end

% Define operators
A  = @(a) my_iudwt(a, f_h, N1, N2);
AS = @(x) my_udwt_adj(x, f_h, N1, N2);
IdPAAS_Inv = @(x) my_IdPAAS_Inv(x, f_i, N1, N2);

[Q1, Q2, Q3] = size(AS(ones(N1, N2)));

op.A = vect(A, Q1, Q2, Q3);
op.AS = vect(AS, N1, N2);

op.IdPAAS_Inv = vect(IdPAAS_Inv, N1, N2);

global silent
silent = ~isempty(silent) && sum(abs(silent)) > 0;
if ~silent
    disp(['Test filter bank']);
end
op = properties_tests(op, Q1 * Q2 * Q3);

function x = my_iudwt(a, f_h, N1, N2)

    C = length(f_h);
    x = zeros(N1, N2);
    for c = 1:C
        k = f_h{c}.k;
        x = x + real(ifft2(f_h{c}.def .* fft2(a(:, (1:N2)+N2*(c-1)))));
    end

function a = my_udwt_adj(x, f_h, N1, N2)

    C = length(f_h);
    a = zeros(N1, C*N2);
    for c = 1:C
        k = f_h{c}.k;
        a(:, (1:N2)+N2*(c-1)) = real(ifft2(f_h{c}.adj .* fft2(x)));
    end

function x = my_IdPAAS_Inv(x, f_i, N1, N2)

    x = real(ifft2(fft2(x) ./ f_i));

function hf = atrous(hf)

    hf = fftshift(hf);

    [N1 N2] = size(hf);
    [cX cY] = fourier_center(N1, N2);
    [X, Y] = meshgrid((1:N2) - cX, (1:N1) - cY);

    if mod(cX, 2) == 0
        rX = 2:2:N1;
    else
        rX = 1:2:N1;
    end
    if mod(cY, 2) == 0
        rY = 2:2:N1;
    else
        rY = 1:2:N1;
    end
    Xmask = (mod(X, 2) == 0);
    Ymask = (mod(Y, 2) == 0);

    hf(rX, rY) = hf(cX + (rX - cX) / 2, cY + (rY - cY) / 2);
    hf(~(Xmask & Ymask)) = 0;

    hf = fftshift(hf);

function [cM, cN] = fourier_center(M,N)

    if mod(M,2) == 1
        cM = (M+3)/2;
    else
        cM = (M+2)/2;
    end
    if mod(N,2) == 1
        cN = (N+3)/2;
    else
        cN = (N+2)/2;
    end
