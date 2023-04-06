# load the forecast models(load, pv) with pt format
# by the model config, train the model with the test data
# calculate the predicted values of pv and load, and then net-load
# calculate the loss


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



now = datetime.datetime.now()
timestamp = now.strftime("%m%d_%H%M")




def plot(actual_netload, calculated_netload, fig_size, font_size, strategy, days, save_path):
    fig = plt.figure(figsize=fig_size)
    length = 10
    actual_netload = actual_netload.detach().numpy()
    calculated_netload = calculated_netload.detach().numpy()
    
    # ax1
    ax1 = plt.subplot(5,1,1)
    i = 0
    plt.title(f'Forecasted Net-load by {strategy} Forecast Strategy', fontsize = font_size)
    ax1.plot(actual_netload[i*length:i*length+length, 0:24].reshape(-1), c='blue', label = 'Actual data')
    ax1.plot(calculated_netload[i*length:i*length+length, 0:24].reshape(-1), c='red', label = 'forecast data')
    x_ticks = []
    x_labels = []
    for j in range(0, 24*(length+1), 24):
        x_ticks.append(j)
        x_labels.append(f'{j//24}')
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels)
    
    # ax2
    ax2 = plt.subplot(5,1,2)
    i = 1
    ax2.plot(actual_netload[i*length:i*length+length, 0:24].reshape(-1), c='blue', label = 'Actual data')
    ax2.plot(calculated_netload[i*length:i*length+length, 0:24].reshape(-1), c='red', label = 'forecast data')
    x_ticks = []
    x_labels = []
    for j in range(0, 24*(length+1), 24):
        x_ticks.append(j)
        x_labels.append(f'{j//24}')
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels)
    
    # ax3 
    ax3 = plt.subplot(5,1,3)
    i = 2
    ax3.plot(actual_netload[i*length:i*length+length, 0:24].reshape(-1), c='blue', label = 'Actual data')
    ax3.plot(calculated_netload[i*length:i*length+length, 0:24].reshape(-1), c='red', label = 'forecast data')
    x_ticks = []
    x_labels = []
    for j in range(0, 24*(length+1), 24):
        x_ticks.append(j)
        x_labels.append(f'{j//24}')
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_labels)
    plt.ylabel('Net-load (kWh)', fontsize = font_size)
    
    # ax4
    ax4 = plt.subplot(5,1,4)
    i = 3
    ax4.plot(actual_netload[i*length:i*length+length, 0:24].reshape(-1), c='blue', label = 'Actual data')
    ax4.plot(calculated_netload[i*length:i*length+length, 0:24].reshape(-1), c='red', label = 'forecast data')
    x_ticks = []
    x_labels = []
    for j in range(0, 24*(length+1), 24):
        x_ticks.append(j)
        x_labels.append(f'{j//24}')
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels(x_labels)
    
    # ax5
    ax5 = plt.subplot(5,1,5)
    i = 4
    ax5.plot(actual_netload[i*length:i*length+length, 0:24].reshape(-1), c='blue', label = 'Actual data')
    ax5.plot(calculated_netload[i*length:i*length+length, 0:24].reshape(-1), c='red', label = 'forecast data')
    x_ticks = []
    x_labels = []
    for j in range(0, 24*(length+1), 24):
        x_ticks.append(j)
        x_labels.append(f'{j//24}')
    ax5.set_xticks(x_ticks)
    ax5.set_xticklabels(x_labels, fontsize = font_size)
    
    plt.legend(loc='lower right', fontsize = font_size)
    plt.xlabel('Day', fontsize = font_size)
    
    # plt.subplots_adjust(bottom=0.1, top=0.5, wspace=0.35)
    plt.savefig(save_path)





def program(version):
    # file to save the results
    f = open(f'./experiment_outputs/COMPARE_RESULTS/{timestamp}_{version}.txt', 'w')
    
    if version == 'RISE':
        # set the file names to use
        # RISE version
        print('RISE version', file = f)
        # load_file_name = "RISE_0329_1701_1_24"
        # pv_file_name = "RISE_0329_1657_1_['SL']_49"
        # net_load_file_name = "RISE_0330_0000_1_49"
        # test_net_load_file_name = "Y_netload_231days_RISE_testset"
        load_file_name = "RISE_0406_1545_1_24"
        pv_file_name = "RISE_0406_1539_1_['SL']_49"
        net_load_file_name = "RISE_0406_1324_1_49"
        test_net_load_file_name = "Y_netload_231days_RISE_testset"
        
    elif version == 'DORM':
        # DORM version
        print('DORM version', file = f)
        # load_file_name = "DORM_0331_1652_1_24"   이게 더 예전 버전
        # load_file_name = "DORM_0331_2229_1_29"
        # pv_file_name = "DORM_0331_1657_1_['SL']_49"
        # net_load_file_name = "DORM_0331_1644_1_49"
        # test_net_load_file_name = "Y_netload_231days_DORM_testset"
        load_file_name = "DORM_0407_0006_1_24"
        pv_file_name = "DORM_0407_0040_1_['SL']_49"
        net_load_file_name = "DORM_0407_0005_1_49"
        test_net_load_file_name = "Y_netload_231days_DORM_testset"
        
        
    elif version == 'MACH':
        print('MACH version', file = f)
        load_file_name = "MACH_0331_1652_1_24"
        pv_file_name = "MACH_0331_1659_1_['SL']_49"
        net_load_file_name = "MACH_0331_1644_1_49"
        test_net_load_file_name = "Y_netload_231days_MACH_testset"



    # load the output files
    load = pd.read_csv(f'./experiment_outputs/test_output/load/{load_file_name}.csv', index_col=0)
    pv = pd.read_csv(f'./experiment_outputs/test_output/pv/{pv_file_name}.csv', index_col=0)
    netload = pd.read_csv(f'./experiment_outputs/test_output/netload/{net_load_file_name}.csv', index_col=0)

    # load the net-load test_y data
    actual_netload = pd.read_csv(f'./processed_data/netload/{test_net_load_file_name}.csv', index_col=0)

    # calculate the predicted net-load
    calculated_netload = load - pv
    
    # convert the data to torch tensor to calculate the loss
    actual_netload_ = torch.FloatTensor(actual_netload.values)
    calculated_netload_ = torch.FloatTensor(calculated_netload.values)
    netload_ = torch.FloatTensor(netload.values)

    # define the indicies for the loss
    mse = nn.MSELoss(reduction = 'mean')
    mae = nn.L1Loss()
    mape = MAPE()


    # INDIRECT model
    indirect_mse = mse(calculated_netload_[:, 0:24], actual_netload_[:, 0:24])
    indirect_mae = mae(calculated_netload_[:, 0:24], actual_netload_[:, 0:24])
    indirect_mape = mape(calculated_netload_[:, 0:24], actual_netload_[:, 0:24])

    # DIRECT model
    direct_mse = mse(netload_[:, 0:24], actual_netload_[:, 0:24])
    direct_mae = mae(netload_[:, 0:24], actual_netload_[:, 0:24])
    direct_mape = mape(netload_[:, 0:24], actual_netload_[:, 0:24])
    
    
    # plot the results
    days = len(actual_netload)
    plot(actual_netload_, calculated_netload_, (50,50), 70, 'Indirect', days, f'./experiment_outputs/COMPARE_RESULTS/{timestamp}_{version}_indirect.png')
    plot(actual_netload_, netload_, (50,50), 70, 'Direct', days, f'./experiment_outputs/COMPARE_RESULTS/{timestamp}_{version}_direct.png')



    print("The Loss of Indirect Model", file = f)
    print('MSE: {:.4f}'.format(indirect_mse), file = f)
    print('MAE: {:.4f}'.format(indirect_mae), file = f)
    print('MAPE(%): {:.4f}'.format(indirect_mape*100), file = f)
    print('-'*50, file = f)
    print('', file = f)



    print("The Loss of Direct Model", file = f)
    print('MSE: {:.4f}'.format(direct_mse), file = f)
    print('MAE: {:.4f}'.format(direct_mae), file = f)
    print('MAPE(%): {:.4f}'.format(direct_mape*100), file = f)
    print('-'*50, file = f)
    print('', file = f)


    f.close()




# program('RISE')
program('DORM')
# program('MACH')