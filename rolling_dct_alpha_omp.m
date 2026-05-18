function [alphaSignal, reconResid] = rolling_dct_alpha_omp(resid, window, K, standardizeWithinWindow)
% Recover transform-sparse alpha using DCT + OMP
%
% Inputs:
%   resid : T x N residual matrix
%   window : rolling window length
%   K : sparsity level for OMP
%
% Outputs:
%   alphaSignal : T x N
%   reconResid  : T x N reconstructed last point

[T, N] = size(resid);
alphaSignal = nan(T, N);
reconResid = nan(T, N);

C = dct_basis_no_toolbox(window);   % orthonormal DCT basis
Ct = C';

for j = 1:N
    xj = resid(:,j);

    for t = window:T
        x = xj(t-window+1:t);

        mu = 0;
        sig = 1;
        if standardizeWithinWindow
            mu = mean(x);
            sig = std(x);
            if sig < 1e-12
                sig = 1;
            end
            xw = (x - mu) / sig;
        else
            xw = x;
        end

        % Solve for sparse DCT coefficients
        % xw = Ct * c, so use dictionary Ct
        chat = omp(Ct, xw(:), K, 1e-6);

        % Reconstruct
        xhat = Ct * chat;

        if standardizeWithinWindow
            xhat = mu + sig * xhat;
        end

        reconResid(t,j) = xhat(end);
        alphaSignal(t,j) = xhat(end);
    end
end
end