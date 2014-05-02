function op = op_haar_udwt_analysis(N1, N2, depth, gamma)
% op_haar_udwt_analysis - create a Daubechies 4 UDWT analysis operator
%
%   op = op_haar_udwt_analysis(N1, N2, depth, gamma)
%
%   N1, N2 are the dimension of the image to analyse.
%   depth is a vector of all levels of the scales.
%   gamma defines the weighting of each scale by 4^(d (gamma - 1)).
%
%   The obtained dictionary is redundant but is not a complete
%   Haar dictionnary. It computes only detail coefficients
%   in vertical and horizontal
%
%   Copyright (c) 2014 Charles Deledalle

mother = [  1,  1 ];
child  = [  1, -1 ];
op = op_wavelet_udwt_analysis(N1, N2, depth, gamma, mother, child);
