%load ./data/AAPL.csv;
T = readtable('AAPL.csv','Format','%{yyyy-mm-dd}D%f%f%f%f%f%f');
warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')
Date = table2array(T(:,1));
Open = table2array(T(:,2));
High = table2array(T(:,3));
Low = table2array(T(:,4));
Close = table2array(T(:,5));
Adj_close = table2array(T(:,6));
Volume = table2array(T(:,7));
length = numel(Open(:,1));

