import pandas as pd


pv = pd.read_csv('./processed_data/pv/week_2021_PV.csv', index_col = 0)
net = pd.read_csv('./processed_data/netload/RISE_week_2021_netload.csv', index_col=0 )

load_index = pv.index.intersection(net.index)

pv = pv.reindex(load_index)
net = net.reindex(load_index)

load = pv + net

# X: 210104-211229, Y: 210105-211230 / 234x24
X = load.iloc[0:234,:]
Y = load.iloc[1:235,:]

load.to_csv("./processed_data/week_2021_load.csv")
