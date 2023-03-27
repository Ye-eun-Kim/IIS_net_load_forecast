import pandas as pd


pv = pd.read_csv('./processed_data/pv/RISE_2021_PV.csv', index_col = 0)
net = pd.read_csv('./processed_data/netload/RISE_all_interpolated.csv', index_col=0 )

net_value = net.iloc[:,0:24]
net_flag = net.iloc[:,24]

load = pv + net_value

load = pd.concat([load, net_flag], axis=1)

load.to_csv("./processed_data/RISE_2021_load.csv")
