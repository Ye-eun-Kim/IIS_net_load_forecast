# Process net-load data without missing data by interpolation


import pandas as pd
import numpy as np


# import calendar file
cal = pd.read_excel('./processed_data/2021_cal_flag.xlsx',index_col=0, header=0)

# extract and combine needed colums from net load excel files into one dataframe temp and then divide them into week and weekend
basic_path = './preprocessed_data/netload_forecasting/GIST energy data/2021 net load/undergraduate_day/학사 일보.gcf_2021-'

# month list of months having only 30 days
thirty_month = [4, 6, 9, 11]

df = pd.DataFrame

# month list of months having only 30 days
thirty_month = [4, 6, 9, 11]

for month in range(1, 13):
    if month == 2:
        end_date = 28
    elif month in thirty_month:
        end_date = 30
    else:
        end_date = 31

    for day in range(1, end_date+1):
        date = '21{0:0>2}{1:0>2}'.format(month, day)

        file_path = f'{basic_path}{month:0>2}-{day:0>2}_23-59.xls'
        
        try:
            # exclude missing data
            temp = pd.read_excel(file_path,  header=[7, 8, 9])
            temp1 = temp["다산빌딩"]["유효전력"]
            temp2 = temp["다산빌딩(E)"]["유효전력"]
            temp = temp1 + temp2
            temp.columns = [date]
            
        except FileNotFoundError:
            print('missing data! / month: ', month, ' date: ', day)
            continue



        # avoid missing data marked with '-' by asytpe method
        try:
            temp.astype('float64')
        except ValueError:
            print('missing data! / month: ', month, ' date: ', day)
            continue

        # need to initialize dataframe 'week' for week and 'weekend' for weekend
        if month == 1 and day == 1:
            df = temp
        else:
            df = pd.concat([df, temp], axis=1)

    print('This month is finished. ', date)
    
df = df.transpose()
df.index = pd.to_numeric(df.index)

# restrict the number of decimal places to 2
df = df.round(1)

df.to_csv("./processed_data/netload/DASAN_2021_netload.csv")