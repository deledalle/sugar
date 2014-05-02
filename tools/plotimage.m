function plotimage(img, L)

% plotimage - plot/display a image
%
%   plotimage(img, L)
%
%   img is the image to display
%   L is teh maximum value (default 255)
%
%   Copyright (c) 2014 Charles Deledalle

    if ~exist('L', 'var');
        L = 255;
    end
    imagesc(img);
    axis image;
    caxis([0 L])
    colormap(gray(256));
    axis off;