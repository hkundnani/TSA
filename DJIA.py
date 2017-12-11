#DJIA preprecoessing. If for a given date, Adj_Close value is missing, use the concave function to fill them.
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:19:03 2017

@author: Jarvis
"""
import sys
import csv
import datetime
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

if (len(sys.argv) < 1):
    print ('You have given wrong number of arguments.')
    print ('Please give arguments in follwing format: test.py input_file_name output_file_name')
else:
    #in_file = sys.argv[1]
    #out_file = sys.argv[2]
    in_file = 'original_djia/MSFT.csv'
    out_file = 'processed_djia/MSFT.csv'
    with open(in_file) as csvfile:
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
            
        #Concave function to fill the missing Adj_close value
        j = 0
        adjCloseList = []
        labellist = []
        for i in range(len(datelist)):
            if datelist[i] == dates[j]:
                adjCloseList.append(str(adj_close[j]))
                j = j + 1
            else:
                adjCloseList.append(str((float(adjCloseList[i-1])+float(adj_close[j]))/2))
                
        #Label based on increase or decrease
        for i in range(len(adjCloseList)):
            if i == 0:
                labellist.extend([1])
                print("Done")
                continue     
            if float(adjCloseList[i])>=float(adjCloseList[i-1]):
                labellist.extend([1])
            else:
                labellist.extend([0])
                       
                

                
        final = zip(datelist,adjCloseList,labellist)
        
     
        #print final[1][1]
        
        
        #Write to file
        with open(out_file,'wb') as outputFile:
            wr = csv.writer(outputFile, dialect='excel')
            wr.writerows(final)
        
