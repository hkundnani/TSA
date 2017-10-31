# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:19:03 2017

@author: Jarvis
"""

import csv
import datetime
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

with open('C:/Users/Jarvis/Desktop/TSA/MSFT.csv') as csvfile:
    next(csvfile)
    readCSV = csv.reader(csvfile)
    dates = []
    adj_close = []
    for row in readCSV:
        dates.append(row[0])
        adj_close.append(row[5])
    #print(dates)
    start_date = '2016-03-01'
    start_date = datetime.strptime(start_date,'%Y-%m-%d')
    end_date = '2016-06-15'
    end_date = datetime.strptime(end_date,'%Y-%m-%d')
    day_count = (end_date - start_date).days + 1
        
    datelist = []
    for single_date in (start_date + timedelta(n) for n in range(day_count)):
        date_string = single_date.strftime('%Y-%m-%d')
        datelist.append(date_string)
        
    j = 0
    adjCloseList = []
    for i in range(len(datelist)):
        if datelist[i] == dates[j]:
            adjCloseList.append(str(adj_close[j]))
            j = j + 1
        else:
            adjCloseList.append(str((float(adjCloseList[i-1])+float(adj_close[j]))/2))
    
    final = zip(datelist,adjCloseList)
    #print final[1][1]
    
with open('Microsoft.csv','wb') as outputFile:
    wr = csv.writer(outputFile, dialect='excel')
    wr.writerows(final)
        
    

#data = pd.read_csv('C:/Users/Jarvis/Desktop/TSA/AAPL.csv',parse_dates=[0])
#data['Date']=pd.to_datetime(data['Date'], format="%Y-%m-%d")
#datelist = pd.date_range('2016-03-01', '2016-06-15',freq='D')
#datelist.format(formatter=lambda x: x.strftime('%Y-%m-%d'))
#data['Adj Close']=pd.to_numeric(data['Adj Close'])
#
#df=pd.DataFrame({'Date':datelist, 'Visited':0, 'Adj Close':0.0})
##df=df.set_index('Date')
#print(len(df))
#
#count = 0
#for index, row in df.iterrows():
#    if row['Date'] == data['Date'].iloc[count]:
#        count = count + 1
#        df[row['Visited']] = 1
        
    


#count = 0
#for i in range(len(df)):
#    if df.Date[i] == data['Date'].iloc[count]:
#        count = count + 1
#        df.loc['Visited'][i]=1  
        #datelist[i][2]=1
        #print datelist[i]
#    else:
#        df.loc['Visited'][i]=0   

#labels = ['Date','Visited','Adj Close']