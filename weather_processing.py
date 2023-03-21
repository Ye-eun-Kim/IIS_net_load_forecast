import pandas as pd
import numpy as np
from datetime import datetime



pv = pd.read_csv('./processed_data/pv/RISE_2021_PV_no_interpol.csv', index_col=0)
forecast_weather = pd.read_csv('./processed_data/weather/forecast_weather.csv')
observation_weather = pd.read_csv('./preprocessed_data/weather_data/weather_observation.csv', index_col=2, encoding='cp949').drop(['지점', '지점명'], axis=1)
load = pd.read_csv('./processed_data/load/week_2021_load.csv', index_col=0)
cal = pd.read_csv('./processed_data/2021_cal_flag.csv', index_col=0)

observation_weather.columns = ['DS', 'SL', 'SR']  # (DS) Duration of Sunlight: 가조시간(hr) | (SL) SunLight 합계 일조시간(hr) | (SR) Solar Radiation 합계 일사량(MJ/m2)
cal_used = cal[['miss&week']]



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


# concat pv, observation_weather, forecast_weather
comb = pd.concat([pv, observation_weather, forecast_weather], axis=1, join='outer')
# 50 features = 24 PV + 3 observation + 23 forecast
#  PV generation each hour
#  'DS', 'SL', 'SR'
#  'TM_6', 'TM_9', 'TM_12', 'TM_15', 'TM_18', 'WS_6', 'WS_9', 'WS_12', 'WS_15', 'WS_18',
#  'SK_6', 'SK_9', 'SK_12', 'SK_5', 'SK_18', 'PP_6', 'PP_9', 'PP_12', 'PP_15', 'PP_18',
#  'PR_9', 'PR_15', 'PR_21'

# sort the index of comb
# after concat, the indices not in intersection are in the end of the dataframe
comb.sort_index(inplace=True)



# set dataframes for pv and load
X_pv = pd.DataFrame(columns=comb.columns)
Y_pv = pd.DataFrame(columns=comb.columns)

X_load = pd.DataFrame(columns=load.columns)
Y_load = pd.DataFrame(columns=load.columns)


# select the data of dates that are valid
# valid: the day and the next day is valid
cnt = 0
for idx, row in comb.iterrows():
    if cal_used.loc[idx]['miss&week'] == 0:
        if cal_used.iloc[cnt+1].values[0] == 0:
            X_pv = X_pv.append(row)
            Y_pv = Y_pv.append(comb.iloc[cnt+1])
            X_load = X_load.append(load.loc[idx])
    cnt+=1
    

Y_load = load.loc[Y_pv.index]


X_pv.to_csv('./processed_data/pv/X_pv.csv')
Y_pv.to_csv('./processed_data/pv/Y_pv.csv')

X_load.to_csv('./processed_data/load/X_load.csv')
Y_load.to_csv('./processed_data/load/Y_load.csv')