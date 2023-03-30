# ex.
# python weather_processing.py RISE
# python weather_processing.py DORM



import pandas as pd
import numpy as np
from datetime import datetime
import sys


def program(building):
    if building == 'RISE':
        pv_path = './processed_data/pv/RISE_2021_PV_no_interpol.csv'
        load_path = './processed_data/load/week_2021_load.csv'
    elif building == 'DORM':
        pv_path = './processed_data/pv/DORM_2021_PV_no_interpol.csv'
        load_path = './processed_data/load/DORM_2021_load.csv'
    elif building == 'DASAN':
        pv_path = './processed_data/pv/DASAN_2021_PV_no_interpol.csv'
        load_path = './processed_data/load/DASAN_2021_load.csv'

    pv = pd.read_csv(pv_path, index_col=0)
    forecast_weather = pd.read_csv('./processed_data/weather/forecast_weather.csv')
    observation_weather = pd.read_csv('./preprocessed_data/weather_data/weather_observation.csv', index_col=2, encoding='cp949').drop(['지점', '지점명'], axis=1)
    load = pd.read_csv(load_path, index_col=0)
    cal = pd.read_csv('./processed_data/2021_cal_flag.csv', index_col=0)

    observation_weather.columns = ['DS', 'SL', 'SR']  # (DS) Duration of Sunlight: 가조시간(hr) | (SL) SunLight 합계 일조시간(hr) | (SR) Solar Radiation 합계 일사량(MJ/m2)
    cal_used = cal[['miss&week', 'miss_flag_pv']]



    # modify date(일시) string into integer to convert the value to tensor.
    # set '일시' as index of observation_weather
    observation_weather_len = len(observation_weather)
    observation_weather_date_list = observation_weather.index
    mdi_observation_weather_date_list = []

    for i in range(observation_weather_len):
        date = observation_weather_date_list[i]
        date_time_obj = datetime.strptime(date, '%Y-%m-%d')
        mdi_date = (date_time_obj.year-2000)*10000 + date_time_obj.month*100 + date_time_obj.day
        mdi_observation_weather_date_list.append(mdi_date)


    # set the modified date as index of observation_weather and forecast_weather    
    observation_weather.index = mdi_observation_weather_date_list
    forecast_weather.index = mdi_observation_weather_date_list

    # rename the column name of forecast_weather from 'SK_5' to 'SK_15'
    forecast_weather = forecast_weather.rename(columns={'SK_5':'SK_15'})

    # concat pv, observation_weather, forecast_weather
    comb = pd.concat([pv, observation_weather, forecast_weather], axis=1, join='outer')
    # 50 features = 24 PV + 3 observation + 23 forecast
    #  PV generation each hour
    #  'DS', 'SL', 'SR'
    #  'TM_6', 'TM_9', 'TM_12', 'TM_15', 'TM_18', 'WS_6', 'WS_9', 'WS_12', 'WS_15', 'WS_18',
    #  'SK_6', 'SK_9', 'SK_12', 'SK_15', 'SK_18', 'PP_6', 'PP_9', 'PP_12', 'PP_15', 'PP_18',
    #  'PR_9', 'PR_15', 'PR_21'

    # sort the index of comb
    # after concat, the indices not in intersection are in the end of the dataframe
    comb.sort_index(inplace=True)



    # set dataframes for pv and load
    X_pv = pd.DataFrame(columns=comb.columns)
    Y_pv = pd.DataFrame(columns=comb.columns)


    # select the data of dates that are valid
    # valid: the day and the next day is valid
    comb_len = len(comb)
    cnt = 0
    for idx, row in comb.iterrows():
        if cnt == comb_len-1:
            break
        if cal_used.iloc[cnt+1]['miss&week'] == 0:
            if cal_used.iloc[cnt]['miss_flag_pv'] == 0:
                X_pv = X_pv.append(row)
                Y_pv = Y_pv.append(comb.iloc[cnt+1])
                # X_load = X_load.append(load.loc[idx])
        cnt+=1
        


    #find intersection of Y_pv and load
    load_pv_y = set(Y_pv.index) & set(load.index)
    drop_index = (set(load.index) - load_pv_y)

    # drop the rows of load that are in drop_index
    load_ = load.drop(drop_index, axis=0)
    X_pv = X_pv.drop(210103, axis=0)
    Y_pv = Y_pv.drop(210104, axis=0)

    len_load = len(load_)

    X_pv.to_csv(f'./processed_data/pv/X_pv_231days_{building}.csv')
    Y_pv.to_csv(f'./processed_data/pv/Y_pv_231days_{building}.csv')
    load_.iloc[0:len_load-1].to_csv(f'./processed_data/load/X_load_231days_{building}.csv')
    load_.iloc[1:len_load].to_csv(f'./processed_data/load/Y_load_231days_{building}.csv')

    load_weather = pd.concat([load_, observation_weather, forecast_weather], axis=1, join='inner')
    load_weather = load_weather.sort_index()
    load_weather.iloc[0:len_load-1].to_csv(f'./processed_data/load/X_load_231days_{building}_weather.csv')
    load_weather.iloc[1:len_load].to_csv(f'./processed_data/load/Y_load_231days_{building}_weather.csv')


    net_path = f'./processed_data/netload/X_netload_231days_{building}.csv'
    net = pd.read_csv(net_path, index_col=0)
    net_ = pd.concat([net, observation_weather, forecast_weather], axis=1, join='inner')
    net_ = net_.sort_index()
    net_.drop(columns = ['SL']).to_csv(f'./processed_data/netload/X_netload_231days_{building}_49features.csv')
    
    
program('RISE')
program('DORM')
program('DASAN')