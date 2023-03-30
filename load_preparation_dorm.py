import pandas as pd


pv = pd.read_csv('./processed_data/pv/DASAN_2021_PV_no_interpol.csv', index_col = 0)
net = pd.read_csv('./processed_data/netload/DASAN_2021_netload.csv', index_col=0 )
load_index_x = pd.read_csv('./processed_data/load/X_load_231days.csv', index_col=0).index
load_index_y = pd.read_csv('./processed_data/load/Y_load_231days.csv', index_col=0).index
load_index = set(load_index_x) | set(load_index_y)



# select only the data in load_index from pv
pv = pv.loc[load_index].sort_index()
net = net.loc[load_index].sort_index()


net_value = net.iloc[:,0:24]

load = pv + net_value

load.to_csv("./processed_data/load/DASAN_2021_load.csv")
