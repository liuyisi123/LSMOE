# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:28:09 2024

@author: Liuyisi
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from moe import MoE
import argparse
from torch.utils.tensorboard import SummaryWriter
from oneD_model.oneD_PCT import PctNet as create_model
from util1 import  train_one_epoch,MyDataset,evaluate,test
import matplotlib.pyplot as plt
from pytorchtools1 import EarlyStopping
from torch.utils.data import  DataLoader
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch_pruning as tp
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# plt.rcParams['font.sans-serif']=['Times new Roman']
#%%
def evaluate_metrics(y_test_last, all_pred_last):
    # Include your metric calculations here
    RMSE = np.sqrt(mean_squared_error(y_test_last, all_pred_last))
    R2 = r2_score(y_test_last, all_pred_last)
    MAE = mean_absolute_error(y_test_last, all_pred_last)
    MSE = mean_squared_error(y_test_last, all_pred_last)

    MEAN = np.mean(y_test_last - all_pred_last)
    SD = np.std(y_test_last - all_pred_last)

    y_dist = np.abs(y_test_last - all_pred_last)
    A = (np.sum(y_dist <= 5)) / len(y_dist)
    B = (np.sum(y_dist <= 10)) / len(y_dist)
    C = (np.sum(y_dist <= 15)) / len(y_dist)

    metrics = {
        'RMSE': RMSE,
        'R2': R2,
        'MAE': MAE,
        'MSE': MSE,
        'MEAN': MEAN,
        'SD': SD,
        'A': A,
        'B': B,
        'C': C
    }

    print('RMSE:', RMSE)
    print('R2:', R2)
    print('MAE:', MAE)
    print('MSE:', MSE)
    print('MEAN:', MEAN)
    print('SD:', SD)
    print('A的比例:', A)
    print('B的比例:', B)
    print('C的比例:', C)

    return metrics

#%%
import torch.nn.init as init
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def main(args):
    expert_counts = [32,64]  # 专家数量列表
    k_values = [5,6,7]  # k值列表
    for experts in expert_counts:
        for k1 in k_values:
            best_losses=[]
            for times in range(3):               
                all_metrics=[]
                device = torch.device(args.device if torch.cuda.is_available() else "cpu")
                print("using {} device.".format(device))
                if os.path.exists("./weights") is False:
                    os.makedirs("./weights")
                tb_writer = SummaryWriter()
            
            #%%Dataset
                wavedata_array = np.load('./data/ppg.npy')
                ppg_data1 = wavedata_array[:,np.newaxis,:]
                first_derivative = np.gradient(ppg_data1, axis=2)
                second_derivative = np.gradient(first_derivative, axis=2)
                normalized_signal = (ppg_data1 - np.min(ppg_data1, axis=2, keepdims=True)) / (np.max(ppg_data1, axis=2, keepdims=True) - np.min(ppg_data1, axis=2, keepdims=True))
                normalized_first_derivative = (first_derivative - np.min(first_derivative, axis=2, keepdims=True)) / (np.max(first_derivative, axis=2, keepdims=True) - np.min(first_derivative, axis=2, keepdims=True))
                normalized_second_derivative = (second_derivative - np.min(second_derivative, axis=2, keepdims=True)) / (np.max(second_derivative, axis=2, keepdims=True) - np.min(second_derivative, axis=2, keepdims=True))    
                wavedata_array = np.concatenate([normalized_signal, normalized_first_derivative, normalized_second_derivative], axis=1)
                wavedata_array = wavedata_array[:,0,:]
                print(wavedata_array.shape)
                cnapindex_data =np.load('./data/cnap.npy')
                labels=np.array(cnapindex_data[:,0:12])#089
                # labels = np.array(cnapindex_data[:, [3, 4, 9]])
                labels=labels[:,np.newaxis]
                print(labels.shape) 
                X_test, X_remaining, y_test, y_remaining = train_test_split(wavedata_array, labels, test_size=0.8, random_state=3047)            
                X_train, X_valid, y_train, y_valid = train_test_split(X_remaining, y_remaining, test_size=0.25)             
                print(X_train.shape)
                print(X_valid.shape)
                print(X_test.shape)
                dataset = MyDataset(X_train, y_train)
                dataset_valid = MyDataset(X_valid, y_valid)
                dataset_test = MyDataset(X_test, y_test)
                train_loader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True)
                val_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True)
                test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
                  
            #%%
                print(experts)
                net = MoE(input_size=640*1, output_size=12, num_experts=experts, hidden_channels=128, noisy_gating=True, k=k1)
                sub_dir = f"experts_{experts}_k_{k1}_times{times}"
                base_save_dir = "./save_data12"
                save_dir = os.path.join(base_save_dir, sub_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            
                net.apply(weights_init) 
                net = net.to(device)
                
                pg = [p for p in net.parameters() if p.requires_grad]
                optimizer = optim.Adam(pg, lr=args.lr, weight_decay=5E-2)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/10)
                net.train()
                val_losses = []
                train_losses=[]
                test_losses=[]
                
                pred_trains=[]
                pred_vals=[]
                pred_tests=[]
                
                label_trains=[]
                label_vals=[]
                label_tests=[]
                label_trains=[]
                label_vals=[]
                best_loss = float('inf')
                patience=25
                temp=np.array([],dtype=np.int64)
                temp1=np.array([],dtype=np.int64)
                temp2=np.array([],dtype=np.int64)
                temp3=np.array([],dtype=np.int64)    
                early_stopping = EarlyStopping(patience=patience, verbose=True)
                # temp2=np.array([],dtype=np.int64)
                for epoch in range(args.epochs):  # loop over the dataset multiple times
                    train_loss, pred_train,label_train= train_one_epoch(model=net,
                                                                    optimizer=optimizer,
                                                                    data_loader=train_loader,
                                                                    device=device,
                                                                    epoch=epoch,
                                                                    temp=temp,
                                                                    temp1=temp1)
                    scheduler.step()
                    # validate
                    val_loss,pred_val,label_val= evaluate(model=net,
                                                          data_loader=val_loader,
                                                          device=device,
                                                          epoch=epoch,
                                                          temp=temp2,
                                                          temp1=temp3)
                    # test
                    test_loss,pred_test,label_test= test(model=net,
                                                          data_loader=test_loader,
                                                          device=device,
                                                          epoch=epoch,
                                                          temp=temp2,
                                                          temp1=temp3)            
                
                    tags = ["train_loss", "val_loss", "learning_rate"]
            
                    pred_trains.append(pred_train)
                    label_trains.append(label_train)
                    pred_vals.append(pred_val)
                    label_vals.append(label_val)
            
                    tb_writer.add_scalar(tags[0], train_loss, epoch)
                    train_losses.append(train_loss)
                    tb_writer.add_scalar(tags[1], val_loss, epoch)
                    val_losses.append(val_loss)
                    tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
                    model= torch.jit.trace(net,torch.Tensor(1,1,640).to(device)).to(device)

                    if best_loss > val_loss:
                        torch.save(net.state_dict(), f"./weights/best_model_expert5_{experts}_k_{k1}_.pth")
                        best_loss = val_loss
                        select_index = epoch
                    early_stopping(val_loss, net)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                best_losses.append(best_loss)
                pred_trains = np.array(pred_trains, dtype=float).T
                label_trains = np.array(label_trains, dtype=float).T
                pred_vals = np.array(pred_vals, dtype=float).T
                label_vals = np.array(label_vals, dtype=float).T
            
                # pred_trains_last = pred_trains[:, -1]
                # label_trains_last = label_trains[:, -1]
                pred_vals_last = pred_vals[:, select_index]
                label_vals_last = label_vals[:, select_index]
                    
                pred_file_path = os.path.join(save_dir, "pred_tests_last.txt")
                label_file_path = os.path.join(save_dir, "label_tests.txt")
                
                with open(pred_file_path, "w") as file:
                    for pred in pred_vals_last:
                        file.write(str(pred) + "\n")
                with open(label_file_path, "w") as file:
                    for label in label_vals_last:
                        file.write(str(label) + "\n")
                        
                val_index = len(train_losses)
                train_index = len(val_losses)
                # plt.plot(range(val_index), val_losses, linewidth=3, label="val_losses")
                # plt.plot(range(train_index), train_losses, linewidth=3, label="train_losses")
                # plt.legend(fontsize=12)
                # plt.xlabel('Epochs', fontsize=18, fontweight='bold')
                # plt.ylabel("Loss", fontsize=18, fontweight='bold')
                # plt.tick_params(labelsize=16)
                # plt.show()    
                output_txt_path = os.path.join(save_dir, "output.txt")
                output_excel_path = os.path.join(save_dir, "output.xlsx")
                try:
                    with open(output_txt_path, "w") as f:
                        f.write("Train Loss\tVal Loss\n")
                        for train_loss, val_loss in zip(train_losses, val_losses):
                            f.write(f"{train_loss}\t{val_loss}\n")
                except IOError as e:
                    print(f"An error occurred while writing to the file: {e}")
                
                # 创建DataFrame
                data = {
                    'Train Loss': train_losses,
                    'Val Loss': val_losses
                }
                df = pd.DataFrame(data)
                print(df.shape)
                try:
                    df.to_excel(output_excel_path, index=False)
                    print('Finished Training')
                except IOError as e:
                    print(f"An error occurred while writing to the Excel file: {e}")
            
                #%%Plot
                all_pred = np.array(pred_vals_last)
                # # all_pred[all_pred>100]=100
                # plt.figure()
                y_test=label_vals_last
                # print(len(y_test))
                y_pred=all_pred
                # plt.scatter(y_test,y_pred)
            
                metrics = evaluate_metrics(y_test, y_pred)
            
                all_metrics.append(metrics)  
                metrics_df = pd.DataFrame(all_metrics)
                metrics_excel_path = os.path.join(save_dir, "valuation_metrics.xlsx")
                metrics_df.to_excel(metrics_excel_path, index=False)
                data1 = {
                    'val_loss': best_losses,
                }
                loss_excel_path = os.path.join(save_dir, "loss.xlsx")
                df1 = pd.DataFrame(data1)
                print(df1.shape)
                try:
                    df1.to_excel(loss_excel_path, index=False)
                    print('Finished Training')
                except IOError as e:
                    print(f"An error occurred while writing to the Excel file: {e}")    
                
                import gc
                del net, optimizer, scheduler  
                gc.collect()  
                torch.cuda.empty_cache() 
                        
        # Add the if __name__ == '__main__': block
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.0001)
    torch.multiprocessing.freeze_support()
    # parser.add_argument('--weights', type=str, default='./checkpoint_train1.pt',
    #                     help='initial weights path')
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)

