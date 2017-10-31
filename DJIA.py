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
        