function [dates, prices, symbols] = load_price_panel(dataFolder)
% LOAD_PRICE_PANEL for Stooq-style TXT files:
% <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>

listing = dir(fullfile(dataFolder, '*.txt'));

if isempty(listing)
    error('No TXT files found in %s', dataFolder);
end

allDates = datetime.empty(0,1);
assetData = struct('symbol', {}, 'dates', {}, 'prices', {});

for i = 1:numel(listing)
    fn = fullfile(listing(i).folder, listing(i).name);
    fprintf('Reading file: %s\n', listing(i).name);

    fid = fopen(fn, 'r');
    if fid == -1
        error('Cannot open file: %s', fn);
    end

    % Skip header line, then parse 10 comma-separated columns
    data = textscan(fid, '%s %s %s %s %f %f %f %f %f %f', ...
        'Delimiter', ',', ...
        'HeaderLines', 1, ...
        'CollectOutput', false);

    fclose(fid);

    % Column 3 = DATE, Column 8 = CLOSE
    rawDates = data{3};
    closePx  = data{8};

    if isempty(rawDates) || isempty(closePx)
        warning('Skipping file %s because it appears empty or malformed.', listing(i).name);
        continue;
    end

    % Convert YYYYMMDD -> datetime
    d = datetime(rawDates, 'InputFormat', 'yyyyMMdd');

    % Remove bad rows if any
    good = ~isnat(d) & ~isnan(closePx);
    d = d(good);
    closePx = closePx(good);

    % Sort by date
    [d, idx] = sort(d);
    closePx = closePx(idx);

    symbol = erase(listing(i).name, '.txt');

    assetData(end+1).symbol = symbol; %#ok<AGROW>
    assetData(end).dates = d;
    assetData(end).prices = closePx;

    allDates = union(allDates, d);
end

if isempty(assetData)
    error('No valid asset files were loaded from %s', dataFolder);
end

dates = allDates(:);
N = numel(assetData);
Tlen = numel(dates);

prices = nan(Tlen, N);
symbols = strings(1, N);

for j = 1:N
    symbols(j) = string(assetData(j).symbol);

    [tf, loc] = ismember(assetData(j).dates, dates);
    prices(loc(tf), j) = assetData(j).prices(tf);
end

% Forward fill then backward fill
for j = 1:N
    x = prices(:,j);

    for t = 2:Tlen
        if isnan(x(t))
            x(t) = x(t-1);
        end
    end

    for t = Tlen-1:-1:1
        if isnan(x(t))
            x(t) = x(t+1);
        end
    end

    prices(:,j) = x;
end
end