import pandas as pd


# import calendar file
cal = pd.read_excel('./processed_data/2021_cal_flag.xlsx',
                    index_col=0, header=0)

# extract and combine needed colums from net load excel files into one dataframe temp and then divide them into week and weekend
basic_path = './preprocessed_data/netload_forecasting/GIST energy data/2021 PV/태양광 일보.gcf_2021-'


# month list of months having only 30 days
thirty_month = [4, 6, 9, 11]


week = pd.DataFrame
weekend = pd.DataFrame

for month in range(1, 13):
    if month == 2:
        end_date = 28
    elif month in thirty_month:
        end_date = 30
    else:
        end_date = 31

    # there isn't data of end of month like 30th or 31st
    for day in range(1, end_date):

        # 1 for weekend and holiday, 0 for else
        flag = 0

        # check if the date is weekend and holiday or not
        date = '21{0:0>2}{1:0>2}'.format(month, day)
        if date == '211224':   # check if the date is 211224(updated on 23.02.24)
            print('missing data! / month: ', month, ' date: ', day)
            continue
        if cal.at[int(date), 'flag'] == 1:
            flag = 1

        file_path = '{0}{1:0>2}-{2:0>2}_.xls'.format(
            basic_path, str(month), str(day))
        try:
            # exclude missing data
            temp = pd.read_excel(file_path,  header=[3, 4, 5]).iloc[0:24, [
                26]]
        except FileNotFoundError:
            print('missing data! / month: ', month, ' date: ', day)
            continue

        # check missing data
        try:
            temp.astype('float64')
        except ValueError:
            print('missing data! / month: ', month, ' date: ', day)
            continue

        # change odd and strange value(ex. -0) to 0
        # I couldn't find a good way to use dataframe, so I converted to numpy for a moment
        temp = temp.to_numpy()
        temp[temp <= 0] = 0
        temp = temp.sum(axis=1)
        temp = pd.DataFrame(temp, index=range(24), columns=[date])

        # need to initialize dataframe 'week' for week and 'weekend' for weekend
        if month == 1 and day == 1:
            weekend = temp
        elif month == 1 and day == 4:
            week = temp
        else:
            if flag == 1:
                weekend = pd.concat([weekend, temp], axis=1)
            else:
                week = pd.concat([week, temp], axis=1)

    print('okay', date)

week = week.transpose()
weekend = weekend.transpose()

week.to_csv("./processed_data/pv/week_2021_PV_sun_to_mon.csv")
weekend.to_csv("./processed_data/pv/weekend_2021_PV_sun_to_mon.csv")
