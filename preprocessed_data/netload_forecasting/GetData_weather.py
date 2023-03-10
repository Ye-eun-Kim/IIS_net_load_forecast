# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:25:00 2019

@author: user
"""
import requests
import os
#from requests import get
import datetime
#import time
import re
#import json
#import scipy.interpolate as spi
import numpy as np
#import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import pandas
from datetime import timedelta

#service_key(from data.go.kr, log-in by sun-me)
service_key= "W4P57DvFf6ROim1S2fh1jle1YG1elp9393E%2Fvauqy1Rbob%2F6ELsHPU3ojEIe1DpTFMPVnZ6c8rM7KKVxP16NfQ%3D%3D"

#download_category
find_category = ["POP", "PTY", "REH", "SKY", "T3H", "UUU", "VVV", "VEC", "WSD", "R06", "S06", "TMN", "TMX"]
find_category_sixhours = ["R06", "S06"]
find_category_interpor = ["POP", "PTY", "REH", "SKY", "T3H", "VEC", "WSD"] #"TMX", "WAV" //  "TMN",

#location
#[광주시첨단1동,성남시상대원동, 서울시성수동, 나주시송월동, 목포시용당동]
loc_x = 56
loc_y = 71

#current_dir = os.getcwd()
current_dir = "/home/rise/Documents/TOC_191031_script/site_B"


def setTestday(daydelta):
    global url, file_time
#define time
    current_time = datetime.datetime.now()+timedelta(days=daydelta)
    file_time = current_time + timedelta(days=1)

    file_time = "%d-%02d-%02d" % (file_time.year, file_time.month, file_time.day)
    base_date = "%d%02d%02d" % (current_time.year, current_time.month, current_time.day)

    print("Forecast Date:%s" % file_time)

    #make url
    url = "http://newsky2.kma.go.kr/service/SecndSrtpdFrcstInfoService2/ForecastSpaceData?serviceKey="+service_key+"&base_date="+base_date+"&base_time=2000&nx="+str(loc_x)+"&ny="+str(loc_y)+"&numOfRows=100&_type=json"
    print(url)


def APIrequest():
    file_name1 = file_time + "_raw"
    file_name2 = file_time + "_15m"

    #API request
    data_kma = requests.get(url)
    data_contents = data_kma.text
    list_data_contents = re.split("\"category\":", data_contents)
    data_arranged={}
    for j in range(0, len(find_category)):
        data_contents_find = []
        for i in range(0, len(list_data_contents)):
            if find_category[j] in list_data_contents[i]:
                data_contents_find.append(list_data_contents[i])
                data_arranged.setdefault(find_category[j], data_contents_find)
    l = []
    df = DataFrame(l)

    for jj in range(0, len(find_category)):
        temp = data_arranged.get(find_category[jj])
    #        print(temp)
        data_value = []
        data_value_temp = []

        if temp:
            for ii in range(0, len(temp)):
                temp_str = temp[ii]
                #arranged by value
                temp_str_start = temp_str.find("\"fcstValue\":") + len("\"fcstValue\":")
                temp_str_end = temp_str.find(",\"nx\"")
                data_value.append(temp_str[temp_str_start:temp_str_end])
            df1 = DataFrame(data_value, columns=[find_category[jj]])
            df = pandas.concat([df, df1], axis=1)
        else:
            pass

    if temp:
        #6_hours data(R06, S06) post-processing
        for counts in range(0, len(find_category_sixhours)):
            df_temp = df[find_category_sixhours[counts]]
            for sixhour_data in range(0, int(len(df_temp)/2)):
                df[find_category_sixhours[counts]][sixhour_data*2] = df_temp[sixhour_data]
            for sixhour_data in range(0, int(len(df_temp)/2)):
                df[find_category_sixhours[counts]][sixhour_data*(2)+1] = None
        df = df.drop(9,0)

        save_dir = current_dir + "/historical_data/weather/rawdata"
        os.chdir(save_dir)

        #make CSV file
        df.to_csv(r'%s.csv' %file_name1, mode='a', header=True, index=False, encoding='cp949')




    #interpolation
        df_fin = DataFrame()
        for index_category in range(0, len(find_category_interpor)):
            data_value_interpor=[]
            for interpor in range(0, len(df[find_category_interpor[index_category]])-1):
                data_value_interpor.append(np.linspace(float(df[find_category_interpor[index_category]][interpor]), float(df[find_category_interpor[index_category]][interpor+1]), 12))
                ll = []
                df_fin_temp = DataFrame(ll)
                for inter_count in range(0, len(data_value_interpor)):
                    temp_data = list(data_value_interpor[inter_count])
                    ll.append(temp_data)
                for list_count in range(0, len(ll)):
                    df_temp = DataFrame(ll[list_count])
                    df_temp.columns = [find_category_interpor[index_category]]
                    df_fin_temp =df_fin_temp.append(df_temp)
            df_fin = pandas.concat([df_fin, df_fin_temp], axis=1)
            #post-processing for integer type
            if find_category_interpor[index_category] in ["VEC", "SKY", "PTY"]:
                df_fin[find_category_interpor[index_category]] = df_fin[find_category_interpor[index_category]].astype('int64')



        save_dir = current_dir + "/historical_data/weather/interp_15min"
        os.chdir(save_dir)

        #make CSV file
        df_fin.to_csv(r'%s.csv' %file_name2, mode='a', header=True, index=False, encoding='cp949')
    else:
        print('No data found')


for daydelta in range(-3, 1):
    setTestday(daydelta)
    APIrequest()
