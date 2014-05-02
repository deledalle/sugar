function res = psnr(hat, star, L)

% psnr - peak signal to noise ratio
%
%   val = psnr(x1, x2, L)
%
%   x1 and x2 are an images.
%   L is a value typically 255 (default)
%   val is the PSNR: 10log10(L^2/MSE)
%
%   Copyright (c) 2014 Charles Deledalle

    if nargin < 3
        L = 255;
    end
    res = 10 * log10(L^2 / mean2((hat - star).^2));

end
