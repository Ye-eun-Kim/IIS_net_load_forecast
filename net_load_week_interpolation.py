# This file interpolates missing data in RISE_week_2021_netload

# import libraries
import pandas as pd
import numpy as np


df = pd.read_excel('./preprocessed_data/netload_forecasting/RISE_week_2021_netload.xlsx', index_col=0, header=0)
cal = pd.read_excel('./processed_data/2021_cal_flag.xlsx', index_col=0, header=0)

