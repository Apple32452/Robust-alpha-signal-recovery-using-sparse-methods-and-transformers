function [alphaSignal, reconResid] = rolling_dct_alpha_lasso(resid, window, lambdaLasso, standardizeWithinWindow)
% Recover transform-sparse alpha using DCT + LASSO (ISTA)
%
% Inputs:
%   resid : T x N residual matrix
%   window : rolling window length
%   lambdaLasso : LASSO penalty
%
% Outputs:
%   alphaSignal : T x N
%   reconResid  : T x N reconstructed last point

[T, N] = size(resid);
alphaSignal = nan(T, N);
reconResid = nan(T, N);

C = dct_basis_no_toolbox(window);
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

        % xw = Ct * c, solve for sparse c using LASSO
        chat = lasso_ista(Ct, xw(:), lambdaLasso, 200, 1e-6);

        xhat = Ct * chat;

        if standardizeWithinWindow
            xhat = mu + sig * xhat;
        end

        reconResid(t,j) = xhat(end);
        alphaSignal(t,j) = xhat(end);
    end
end