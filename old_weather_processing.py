# Process weather data with observation and forecast and PV data
# data duration: 360days 210101~211231 (210618, 210930, 211222~211224 missing)

# 50 features = 24 PV + 3 observation + 23 forecast
#  PV generation each hour
#  'DS', 'SL', 'SR'
'''
'TM_6', 'TM_9', 'TM_12', 'TM_15', 'TM_18', 'WS_6', 'WS_9', 'WS_12', 'WS_15', 'WS_18',
'SK_6', 'SK_9', 'SK_12', 'SK_5', 'SK_18', 'PP_6', 'PP_9', 'PP_12', 'PP_15', 'PP_18',
'PR_9', 'PR_15', 'PR_21'
'''


# import libraries
import pandas as pd
from datetime import datetime


# load files
pv = pd.read_csv('./processed_data/pv/RISE_2021_PV_no_interpol.csv', index_col=0)
forecast_weather = pd.read_csv('./processed_data/weather/forecast_weather.csv')  # 기온(°C), 풍속(m/s), 하늘상태(1,3,4), 강수확률(%), 강수량(mm)
observation_weather = pd.read_csv('./preprocessed_data/weather_data/weather_observation.csv', index_col=2, encoding='cp949').drop(['지점', '지점명'], axis=1)
observation_weather.columns = ['DS', 'SL', 'SR']  # (DS) Duration of Sunshine: 가조시간(hr) | (SL) SunLight 합계 일조시간(hr) | (SR) Solar Radiation 합계 일사량(MJ/m2)


# modify date(일시) string into integer to convert the value to tensor.
observation_weather_len = len(observation_weather)
observation_weather_date_list = observation_weather.index
mdi_observation_weather_date_list = []

for i in range(observation_weather_len):
    date = observation_weather_date_list[i]
    date_time_obj = datetime.strptime(date, '%Y-%m-%d')
    mdi_date = (date_time_obj.year-2000)*10000 + date_time_obj.month*100 + date_time_obj.day
    mdi_observation_weather_date_list.append(mdi_date)


# set date as index of weather data
observation_weather.index = mdi_observation_weather_date_list
forecast_weather.index = mdi_observation_weather_date_list


# make combined dataframe
# 50 features = 24 PV + 3 observation + 23 forecast
comb = pd.concat([pv, observation_weather, forecast_weather], axis=1, join='inner')

# There are NaN values in 210618, 210930
comb = comb.drop([210618, 210930], axis=0)

# save the dataframe into file
comb.to_csv('./processed_data/pv/2021_weather_and_RISEPV.csv')