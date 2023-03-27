# Process pv data without missing data by interpolation


import pandas as pd
import numpy as np


# import calendar file
cal = pd.read_excel('./processed_data/2021_cal_flag.xlsx', index_col=0, header=0)

# extract and combine needed colums from net load excel files into one dataframe temp and then divide them into week and weekend
basic_path = './preprocessed_data/netload_forecasting/GIST energy data/2021 PV/태양광 일보.gcf_2021-'


# month list of months having only 30 days
thirty_month = [4, 6, 9, 11]

df = pd.DataFrame


for month in range(1, 13):
    if month == 2:
        end_date = 28
    elif month in thirty_month:
        end_date = 30
    else:
        end_date = 31

    # there isn't data of end of month like 30th or 31st
    for day in range(1, end_date+1):

        # check if the date is weekend and holiday or not
        date = f'21{month:0>2}{day:0>2}'
        file_path = f'{basic_path}{month:0>2}-{day:0>2}_.xls'
        
        try:
            # exclude missing data
            temp = pd.read_excel(file_path,  header=[3, 4, 5]).iloc[0:24, [44]]  # 26: RISE, 44: undergrad dorm B
        except FileNotFoundError:
            # linear interpolation to fill missing data
            # read data as NaN for missing data
            temp = pd.DataFrame(index=range(24), columns=[date]).astype(float)
            temp[:] = np.nan
            print('missing data! / month: ', month, ' date: ', day)
            # continue

        if date == '211224':   # check if the date is 211224(updated on 23.02.24)
            temp = pd.DataFrame(index=range(24), columns=[date]).astype(float)
            temp[:] = np.nan
            print('missing data! / month: ', month, ' date: ', day)
                
        temp.reindex(range(24))
        temp.columns = [date]

        if month == 1 and day == 1:
            df = temp
        else:
            df = pd.concat([df, temp], axis=1)

    print('This month is finished. ', date)
    
            
# perform linear interpolation
df = df.transpose()
df.index = pd.to_numeric(df.index)
df = df.interpolate(method='akima', axis=0)

# change negative value to 0 in df because the interpolated values are not all positive or zero
df[df < 0] = 0
# restrict the number of decimal places to 2
df = df.round(1)

df.to_csv("./processed_data/pv/RISE_2021_PV.csv")
