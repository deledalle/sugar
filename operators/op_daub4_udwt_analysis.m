function op = op_daub4_udwt_analysis(N1, N2, depth, gamma)

% op_daub4_udwt_analysis - create a Daubechies 4 UDWT analysis operator
%
%   op = op_daub4_udwt_analysis(N1, N2, depth, gamma)
%
%   N1, N2 are the dimension of the image to analyse.
%   depth is a vector of all levels of the scales.
%   gamma defines the weighting of each scale by 4^(d (gamma - 1)).
%
%   The obtained dictionary is redundant but is not a complete
%   Daubechies 4 dictionnary. It computes only detail coefficients
%   in vertical and horizontal
%
%   Copyright (c) 2014 Charles Deledalle

mother = [  0.6830127,  1.1830127, 0.3169873, -0.1830127 ];
child  = [ -0.1830127, -0.3169873, 1.1830127, -0.6830127 ];
op = op_wavelet_udwt_analysis(N1, N2, depth, gamma, mother, child);
