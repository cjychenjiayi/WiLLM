function y = hampel(x, k, t)
%HAMPEL  Hampel identifier for outliers (MATLAB-like implementation)
%   y = hampel(x,k,t) removes outliers from vector x by replacing
%   values that deviate from the local median by more than t*MAD
%   within a sliding window of radius k. If k is scalar and >1,
%   window is taken as +-k around each point. If omitted, k=3, t=3.

if nargin < 2 || isempty(k)
    k = 3;
end
if nargin < 3 || isempty(t)
    t = 3;
end

% Support vectors and matrices (operate along columns)
sz = size(x);
if isvector(x)
    x = x(:); % column vector
    y = x;
    n = numel(x);
    for i = 1:n
        L = max(1, i - k);
        R = min(n, i + k);
        w = x(L:R);
        m = median(w);
        mad = median(abs(w - m));
        sigma = 1.4826 * mad; % convert MAD to robust estimate of std
        if sigma == 0
            isout = abs(x(i) - m) > (t * eps);
        else
            isout = abs(x(i) - m) > (t * sigma);
        end
        if isout
            y(i) = m;
        end
    end
    % preserve original orientation
    if sz(1) == 1
        y = y.';
    end
else
    % matrix: apply column-wise
    y = x;
    for col = 1:sz(2)
        colv = x(:,col);
        n = numel(colv);
        for i = 1:n
            L = max(1, i - k);
            R = min(n, i + k);
            w = colv(L:R);
            m = median(w);
            mad = median(abs(w - m));
            sigma = 1.4826 * mad;
            if sigma == 0
                isout = abs(colv(i) - m) > (t * eps);
            else
                isout = abs(colv(i) - m) > (t * sigma);
            end
            if isout
                y(i,col) = m;
            end
        end
    end
end
end
