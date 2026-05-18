function rets = compute_log_returns(prices)
rets = diff(log(prices));
end
