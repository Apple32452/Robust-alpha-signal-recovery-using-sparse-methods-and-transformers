function make_synthetic_price_data(dataFolder)
if ~isfolder(dataFolder), mkdir(dataFolder); end
symbols = ["AAA","BBB","CCC","DDD","EEE","FFF","GGG","HHH","III","JJJ"];
dates = (datetime(2020,1,1):caldays(1):datetime(2022,12,31))';
dates = dates(weekday(dates) ~= 1 & weekday(dates) ~= 7);
T = numel(dates);
mkt = 0.0002 + 0.01 * randn(T,1);
for j = 1:numel(symbols)
    beta = 0.5 + rand(); idio = 0.012 * randn(T,1);
    alpha = zeros(T,1);
    shockIdx = randperm(T, round(0.02*T));
    alpha(shockIdx) = 0.03 * sign(randn(numel(shockIdx),1));
    alpha = alpha + 0.001 * sin((1:T)' / (5 + 2*j));
    rets = beta * mkt + idio + alpha;
    price = 100 * exp(cumsum(rets));
    TT = table(dates, price, 'VariableNames', {'Date','Adj_Close'});
    TT.Properties.VariableNames{'Adj_Close'} = 'Adj Close';
    writetable(TT, fullfile(dataFolder, symbols(j) + ".csv"));
end
end
