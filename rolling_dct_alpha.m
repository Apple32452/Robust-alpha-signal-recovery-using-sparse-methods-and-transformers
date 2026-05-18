function [alphaSignal, reconResid] = rolling_dct_alpha(resid, window, lambda, numDctKeep, standardizeWithinWindow)
[T, N] = size(resid);
alphaSignal = nan(T, N); reconResid = nan(T, N);
C = dct_basis_no_toolbox(window); Ct = C';
for j = 1:N
    xj = resid(:,j);
    for t = window:T
        x = xj(t-window+1:t);
        mu = 0; sig = 1;
        if standardizeWithinWindow
            mu = mean(x); sig = std(x); if sig < 1e-12, sig = 1; end
            xw = (x - mu) / sig;
        else
            xw = x;
        end
        c = C * xw;
        if ~isempty(numDctKeep) && numDctKeep < window
            mask = false(window,1); mask(1:numDctKeep) = true; c(~mask) = 0;
        end
        chat = soft_threshold(c, lambda);
        xhat = Ct * chat;
        if standardizeWithinWindow, xhat = mu + sig * xhat; end
        reconResid(t,j) = xhat(end);
        alphaSignal(t,j) = xhat(end);
    end
end
end
