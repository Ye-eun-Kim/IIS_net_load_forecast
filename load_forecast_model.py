# Need two additional arguments: case
# ex.
# python load_forecast_model.py 1 25 "./processed_data/RISE_2021_load.csv"    ---> deleted
# python load_forecast_model.py 1


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
import datetime
import sys
import Dataset_Class as DC


# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DEVICE = 'cpu'
RANDOM_SEED = 2023
EPOCHS = 50000
LEARNING_RATE = 0.000001
BATCH_SIZE = 16


class Net(nn.Module):
    def __init__(self, num_of_features, **model_config):
        super(Net, self).__init__()
        self.model_type = model_config['case']
        if model_config['case'] == 1:
            self.hidd_dim = 256   # good
            self.hidden_dim = 36
        elif model_config['case'] == 2:
            self.hidd_dim = 1024
            self.hidden_dim = 8192
            self.hidden_dimm = 144
        elif model_config['case'] == 3:
            self.hidd_dim = 240
            self.hidden_dim = 72
        elif model_config['case'] == 4:
            self.hidd_dim = 120
            self.hidden_dim = 36
            
        self.fc1 = nn.Linear(num_of_features, self.hidd_dim)
        self.fc2 = nn.Linear(self.hidd_dim, self.hidden_dim)
        # self.fc3 = nn.Linear(self.hidden_dim, 24)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dimm)
        self.fc4 = nn.Linear(self.hidden_dimm, 24)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        # output = self.fc3(x)
        x = self.fc3(x)
        x = self.relu(x)
        output = self.fc4(x)
        return output
    

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


# load the data and split into X and Y
def load_data():
    X_load = pd.read_csv('./processed_data/load/X_load.csv', index_col=0)
    Y_load = pd.read_csv('./processed_data/load/Y_load.csv', index_col=0)
    label_interval = get_label_interval(X_load)
    X = torch.FloatTensor(X_load.values)
    Y = torch.FloatTensor(Y_load.values)
    return X, Y, label_interval


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


# get the number of days of each month as label_interval list
def get_label_interval(X):
    label_interval = []
    for i in range(1, 13):
        cnt = 0
        for idx in X.index:
            if '21{0:0>2}'.format(i) in str(idx):
                cnt+=1
        label_interval.append(cnt)



# plot a daily load sum graph
def plot_daily_load(X, label_interval, fig_size, title, font_size, mini_train_point, valid_point, save_path):
    # sum up the load of each day
    data = torch.sum(X[:,0:24], axis=1).detach().numpy().reshape(-1)
    
    # Create figure and plot the data
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes()
    ax.plot(data)

    plt.title(title, fontsize = font_size)
    plt.xlabel('Month', fontsize = font_size)
    plt.ylabel('Daily load sum (kWh)', fontsize = font_size)

    # Set the x-tick positions and labels
    x_ticks = []
    x_labels = []
    for i, interval in enumerate(label_interval):
        start = sum(label_interval[:i])            # X를 그대로 사용하지 않고, 일자별로 load 합계를 구했기 때문에 feature 개수인 24를 곱하지 않는다.
        x_ticks.append(start)
        x_labels.append(f'{i+1}')

    plt.axvline(x = mini_train_point, c='r')
    plt.axvline(x = valid_point, c='r')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    # save the figure
    plt.savefig(save_path)


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
    plt.ylabel('Load', fontsize = font_size)
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
num_of_features = 24
mae = nn.L1Loss()
mape = MAPE()


# data loading
# X: 210104-211229, Y: 210105-211230 / 173*50
X, Y, label_interval = load_data()
dataset = DC.CustomDataset(X, Y)
data_len = len(dataset)
mini_train_dataloader, valid_dataloader, train_dataloader, test_dataloader, mini_train_size, train_size =  split_data(X, Y, BATCH_SIZE, data_len, 0.8, 0.8)
# 110개   210104 ~ 210816
# 28개    210817 ~ 211013
# 138개   210104 ~ 211013
# 35개    211014 ~ 211229


# load plotting
dir = './experiment_outputs/load_forecast/'
now = datetime.datetime.now()
timestamp = now.strftime("%m%d_%H%M")

file_name = f'{model_case}_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}_{timestamp}'
plot_daily_load(X, label_interval, (10,4), "Daily Load Sum in 2021", 8, mini_train_size-1, train_size-1, dir+"plots/daily_load/"+file_name+'.png')


# model setting
model_config = {'case': model_case}
model = Net(num_of_features, **model_config).to(DEVICE)
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

plot_loss(mini_train_loss_arr, val_loss_arr, range_start=20, best_val_epoch=best_val_epoch+3, fig_size=(10,6), title = 'Training Performance of the Model', font_size = 10, save_path = dir+f'plots/loss/{file_name}.png')



# training and testing
set_seed(RANDOM_SEED)

for epoch in range(best_val_epoch):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    if epoch % 500 == 0:
        print(f'Train Epoch: {epoch:4d}/{best_val_epoch}  |  Train Loss {train_loss:.6f}', file = f)

test_output, test_y = evaluate(model, test_dataloader)

test_mse = criterion2(test_output[:, 0:24], test_y[:, 0:24])
test_mae = mae(test_output[:, 0:24], test_y[:, 0:24])
test_mape = mape(test_output[:, 0:24], test_y[:, 0:24])

print('Test Loss', file = f)
print('MSE: {:.6f}'.format(test_mse), file = f)
print('MAE: {:.6f}'.format(test_mae), file = f)
print('MAPE(%): {:.6f}'.format(test_mape*100), file = f)



plot(1, 20, test_output[:, 0:24], test_y[:, 0:24], (20, 5), 'Actual and forecast load for 20 days', 18, dir+f'plots/forecasted_load/{file_name}.png')



f.close()

torch.save(model.state_dict(), dir+f'models/{file_name}.pt')