# 230321


# TODO: SK_15여야 하는데 SK_5로 되어있음. 수정 필요

# Need two additional arguments: case, drop_features(list)
# ex.
# python pv_forecast_model.py 1 PR_9,PR_15,PR_21 -- X
# python pv_forecast_model.py 1 PP_6,PP_9,PP_12,PP_15,PP_18 -- X
# python pv_forecast_model.py 1 TM_6,TM_9,TM_12,TM_15,TM_18 -- X
# python pv_forecast_model.py 1 WS_6,WS_9,WS_12,WS_15,WS_18 -- X
# python pv_forecast_model.py 1 SK_6,SK_9,SK_12,SK_15,SK_18 -- X
# python pv_forecast_model.py 1 DS -- X
# python pv_forecast_model.py 1 SL -- X
# python pv_forecast_model.py 1 SR -- X

# python pv_forecast_model.py 1

# all the features in pv data
'''
['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
'13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', 'DS',
'SL', 'SR', 'TM_6', 'TM_9', 'TM_12', 'TM_15', 'TM_18', 'WS_6', 'WS_9',
'WS_12', 'WS_15', 'WS_18', 'SK_6', 'SK_9', 'SK_12', 'SK_15', 'SK_18',
'PP_6', 'PP_9', 'PP_12', 'PP_15', 'PP_18', 'PR_9', 'PR_15', 'PR_21']
'''



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsolutePercentageError as MAPE
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import datetime, sys, os
import Dataset_Class as DC



# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DEVICE = 'cpu'
RANDOM_SEED = 2023
EPOCHS = 50000
LEARNING_RATE = 0.000001
BATCH_SIZE = 16
# BATCH_SIZE = int(sys.argv[2])

# patience 3, batch_size 16, learning rate 0.000001, model(50-64-256-64-24) shows the best result



class Net(nn.Module):
    def __init__(self, col_len, **model_config):
        super(Net, self).__init__()
        self.model_type = model_config['case']
        if model_config['case'] == 1:
            self.hidd_dim = 512
            self.hidden_dim = 75
        elif model_config['case'] == 2:
            self.hidd_dim = int(col_len*10)
            self.hidden_dim = int(col_len*10*0.2*0.3)
        elif model_config['case'] == 3:
            self.hidd_dim = 500
            self.hidden_dim = 75  #good
        elif model_config['case'] == 4:
            self.hidd_dim = 250
            self.hidden_dim = 75
            
        self.fc1 = nn.Linear(col_len, self.hidd_dim)
        self.fc2 = nn.Linear(self.hidd_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 24)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.fc3(x)
        return output



# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# load the data and split into X and Y
def load_data(drop_features):
    X_pv = pd.read_csv('./processed_data/pv/X_pv.csv', index_col=0)
    Y_pv = pd.read_csv('./processed_data/pv/Y_pv.csv', index_col=0)
    if drop_features != 'No_drop':
        X = X_pv.drop(columns = drop_features)
    else:
        X = X_pv
        
    Y = Y_pv.iloc[:,0:24]
    col = X.columns
    col_len = len(col)
    X = torch.FloatTensor(X.values)
    Y = torch.FloatTensor(Y.values)
    return X, Y, col, col_len


# split data into mini_train, valid, train, test
def split_data(X, Y, batch_size, data_len, train_pie, mini_train_pie):
    train_size = int(data_len * train_pie)
    mini_train_size = int(train_size * mini_train_pie)
    
    train_data = DC.CustomDataset(X[:train_size], Y[:train_size])
    test_data = DC.CustomDataset(X[train_size:], Y[train_size:])
    mini_train_data = DC.CustomDataset(X[:mini_train_size], Y[:mini_train_size])
    valid_data = DC.CustomDataset(X[mini_train_size:train_size], Y[mini_train_size:train_size])
    mini_train_dataloader = DataLoader(mini_train_data, batch_size = batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_data, batch_size = len(valid_data), shuffle = False)
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = len(test_data), shuffle = False)
    
    return mini_train_dataloader, valid_dataloader, train_dataloader, test_dataloader, mini_train_size, train_size


# create folder if not exist
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


# plot daily feature graphs
def plot_daily_feature(X, label_interval, col_list, fig_size, font_size, mini_train_point, valid_point, pre_save_path):
    unit=''
    pv_col_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
    for idx, col in enumerate(col_list):
        if col in pv_col_list:
            if col == '0':
                data = torch.sum(X[:,idx:idx+24], axis=1).detach().numpy().reshape(-1)
                col = 'PV Generation'
                unit = 'kWh'
            else:
                continue
        elif 'DS' in col:
            # column: duration of sunshine
            data = X[:,idx]
            unit = 'hr'
        elif 'SL' in col:
            # column: sunlight
            data = X[:,idx]
            unit = 'hr'
        elif 'SR' in col:
            # column: solar radiation
            data = X[:,idx]
            unit = 'MJ/m2'
        elif 'TM_15' in col:
            #column: temperature of 15h on next day
            data = X[:,idx]
            unit = '℃'
        elif 'WS_15' in col:
            # column: wind speed of 15h on next day
            data = X[:,idx]
            unit = 'm/s'
        elif 'SK_15' in col:
            # column: state of sky of 15h on next day
            data = X[:,idx]
            col = 'SK_15'
            unit = '(1: clear, 3: partly cloudy, 4: cloudy)' 
        elif 'PP_15' in col:
            # column: probability of precipitation of 15h on next day
            data = X[:,idx]
            unit = '%'
        elif 'PR_15' in col:
            #column: precipitation of 15h on next day
            data = X[:,idx]
            unit = 'mm'
        else:
            continue

        # Create figure and plot the data
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes()
        ax.plot(data)

        plt.title(col+" Fluctuation in 2021", fontsize = font_size)
        plt.xlabel('Month', fontsize = font_size)
        plt.ylabel(col+' ('+unit+')', fontsize = font_size)

        # Set the x-tick positions and labels
        x_ticks = []
        x_labels = []
        for i, interval in enumerate(label_interval):
            start = sum(label_interval[:i])
            x_ticks.append(start)
            x_labels.append(f'{i+1}')

        plt.axvline(x = mini_train_point, c='r')
        plt.axvline(x = valid_point, c='r')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        
        # save the figure
        plt.savefig(pre_save_path+col+'.png')


def plot_loss(mini_train_loss_arr, val_loss_arr, range_start, best_val_epoch, fig_size, title, font_size, save_path):
    fig = plt.figure(figsize=fig_size)
    
    plt.subplot(221)
    plt.title(title, fontsize = font_size)
    # plt.xlabel('Epoch', fontsize = font_size)
    plt.ylabel('Loss(MSE)', fontsize = font_size)
    plt.plot(mini_train_loss_arr, c = 'blue', label = 'Train')
    plt.plot(val_loss_arr, c = 'orange', label = 'Validation')
    plt.legend(loc='upper right', fontsize = font_size)

    plt.subplot(222)
    plt.title(title+' from Epoch '+str(range_start)+' every epoch 200', fontsize = font_size)
    # plt.xlabel('Epoch', fontsize = font_size)
    plt.ylabel('Loss(MSE)', fontsize = font_size)
    plt.plot(np.arange(range_start, best_val_epoch, 200), mini_train_loss_arr[range_start:best_val_epoch:200], c = 'blue', label = 'Train')
    plt.plot(np.arange(range_start, best_val_epoch, 200), val_loss_arr[range_start:best_val_epoch:200], c = 'orange', label = 'Validation')
    plt.legend(loc='upper right', fontsize = font_size)

    plt.subplot(223)
    plt.title(title+' from Epoch '+str(range_start), fontsize = font_size)
    plt.xlabel('Epoch', fontsize = font_size)
    plt.ylabel('Loss(MSE)', fontsize = font_size)
    plt.plot(mini_train_loss_arr[range_start:], c = 'blue', label = 'Train')
    plt.legend(loc='upper right', fontsize = font_size)

    plt.subplot(224)
    plt.title(title+' from Epoch '+str(range_start), fontsize = font_size)
    plt.xlabel('Epoch', fontsize = font_size)
    plt.ylabel('Loss(MSE)', fontsize = font_size)
    plt.plot(val_loss_arr[range_start:], c = 'orange', label = 'Validation')
    plt.legend(loc='upper right', fontsize = font_size)

    plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.35, hspace=0.35)
    
    # save the figure
    plt.savefig(save_path)

    
def plot(i, length, output, Y, fig_size, title, font_size, save_path):
    fig = plt.figure(figsize=fig_size)
    plt.title(title, fontsize = font_size)
    plt.xlabel('Time (h)', fontsize = font_size)
    plt.ylabel('PV', fontsize = font_size)
    plt.plot(Y.detach().numpy()[i:i+length,:].reshape(-1), c='blue', label = 'Actual data')
    plt.plot(output.detach().numpy()[i:i+length,:].reshape(-1), c='red', label = 'forecast data')
    plt.legend(loc='lower right', fontsize = 13)
    plt.savefig(save_path)

     
def train(model, train_dataloader, optimizer, criterion):
    model.train()
    loss_sum = 0.0
    for (x, y) in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        output = model(x)
        train_loss = criterion(output, y)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        loss_sum += train_loss
    
    return (loss_sum/len(train_dataloader.dataset)/24).item() # loss of each epoch


def evaluate(model, valid_dataloader):
    model.eval()
    with torch.no_grad():
        for (x, y) in valid_dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
                
            output = model(x)
            
        return output, y


set_seed(RANDOM_SEED)
model_case = int(sys.argv[1])
if len(sys.argv) == 3:
    drop_features = sys.argv[2].split(',')
else:
    drop_features = 'No_drop'
mae = nn.L1Loss()
mape = MAPE()



# data loading
# X: 210104-211229, Y: 210105-211230 / 175*50
X, Y, col_list, col_len = load_data(drop_features)
dataset = DC.CustomDataset(X, Y)
data_len = len(dataset)
mini_train_dataloader, valid_dataloader, train_dataloader, test_dataloader, mini_train_size, train_size =  split_data(X, Y, BATCH_SIZE, data_len, 0.8, 0.8)
# 112개   210104 ~ 210818
# 28개    210819 ~ 211018
# 140개   210104 ~ 211018
# 35개    211019 ~ 211229



# data plotting
dir = './experiment_outputs/pv_forecast/'
now = datetime.datetime.now()
timestamp = now.strftime("%m%d_%H%M")
label_interval = [13, 13, 17, 16, 14, 11, 17, 16, 16, 13, 14, 13]

file_name = f'{model_case}_{drop_features}_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}_{timestamp}'
_path = dir+"plots/daily_pv_features/"+file_name
create_folder(_path)
plot_daily_feature(X, label_interval, col_list, (10,2.5), 8, mini_train_size-1, train_size-1, _path+'/')


# model setting
model_config = {'case': model_case}
model = Net(col_len, **model_config).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.MSELoss(reduction='sum').to(DEVICE)
criterion2 = nn.MSELoss(reduction = 'mean').to(DEVICE)


# file to save the results
f = open(dir+f"results/{file_name}.txt", 'w')


# model training and validation
mini_train_loss_arr = []
val_loss_arr = []

best_val_loss = float('inf')
best_val_epoch = 0
patience = 0

for epoch in range(EPOCHS):
    mini_train_loss = train(model, mini_train_dataloader, optimizer, criterion)
    mini_train_loss_arr.append(mini_train_loss)
    val_output, val_y = evaluate(model, valid_dataloader)
    val_loss = criterion2(val_output, val_y)
    val_loss_arr.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = epoch
        patience = 0
    # early stopping
    else:
        patience+=1
        if patience > 1:
            print(f'Patience is increased, patience: {patience}', file = f)
    if epoch % 500 == 0:
        # write a log on file
        print(f'Train Epoch: {epoch:4d}/{EPOCHS}  |  Train Loss {mini_train_loss:.6f}  |  Val Loss {val_loss:.6f}', file = f)
    if patience == 3:
        break
        
print('-'*80, file = f)
print(f'The Best Epoch: {best_val_epoch}  |  The Best Validation Error: {best_val_loss:.6f}', file = f)
print('-'*80, file = f)
print('-'*80, file = f)

plot_loss(mini_train_loss_arr, val_loss_arr, range_start=20, best_val_epoch = best_val_epoch+3, fig_size=(10,6), title = 'Training Performance of the Model', font_size = 10, save_path = dir+f'plots/loss/{file_name}.png')



# training and testing
set_seed(RANDOM_SEED)

for epoch in range(best_val_epoch):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    if epoch % 500 == 0:
        print(f'Train Epoch: {epoch:4d}/{best_val_epoch}  |  Train Loss {train_loss:.6f}', file = f)

test_output, test_y = evaluate(model, test_dataloader)
# post processing of the output
# PV generation cannot be negative & PV generation occurs only for 6-20h
test_output = torch.where(test_output > 0, test_output, 0)
test_output[:,0:6] = 0
test_output[:, 21:] = 0


test_mse = criterion2(test_output[:, 0:24], test_y[:, 0:24])
test_mae = mae(test_output[:, 0:24], test_y[:, 0:24])    
# cannot use mape...
# test_mape = mape(test_output[:, 0:24], test_y[:, 0:24])


print('Test Loss', file = f)
print('MSE: {:.6f}'.format(test_mse), file = f)
print('MAE: {:.6f}'.format(test_mae), file = f)
# print('MAPE(%): {:.6f}'.format(test_mape*100), file = f)



plot(1, 20, test_output[:, 0:24], test_y[:, 0:24], (20, 5), 'Actual and forecast PV for 20 days', 18, dir+f'plots/forecasted_pv/{file_name}.png')



f.close()

torch.save(model.state_dict(), dir+f'models/{file_name}.pt')