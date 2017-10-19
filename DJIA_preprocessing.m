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

%preproccessing Date
D=datevec(Date);
D(:,2)=D(:,5);
D(:,5)=zeros;
Date=datetime(D);
startdate = Date(1,1);
enddate = Date(length,1);
filledDate = startdate:enddate;
filledDate = filledDate'; %transpose - convert row to column matrix

count = 1;
for i=1:numel(filledDate(:,1))
    if (filledDate(i,1) == Date(count,1))
        visited(i,1) = 1;
        visited(i,2) = Adj_close(count,1);
		count = count + 1;
	else
		visited(i,2) = -1;
        Adj1 = visited(i-1,2);
		Adj2 = Adj_close(count,1);
		avg = (Adj1 + Adj2)/2;
		visited(i,2) = avg;
    end
end

filledDate.Format = 'yyyy-MM-dd';
newdata = [num2cell(visited) cellstr(filledDate)];
dlmcell('AAPL',newdata,'\t');

%T1 = readtable('AAPL');

