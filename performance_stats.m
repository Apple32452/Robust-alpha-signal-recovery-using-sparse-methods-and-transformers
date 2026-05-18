function stats = performance_stats(rets, annualization)
rets = rets(:); rets = rets(~isnan(rets));
mu = mean(rets); sig = std(rets);
if sig < 1e-12, sharpe = 0; else, sharpe = sqrt(annualization) * mu / sig; end
cumCurve = cumprod(1 + rets);
runningMax = cummax(cumCurve);
drawdown = cumCurve ./ runningMax - 1;
maxDD = min(drawdown);
n = numel(rets);
cagr = cumCurve(end)^(annualization / max(n,1)) - 1;
stats = table(mu, sig, sharpe, cagr, maxDD, 'VariableNames', {'MeanDailyRet','StdDailyRet','Sharpe','CAGR','MaxDrawdown'});
end
