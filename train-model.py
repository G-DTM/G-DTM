from __future__ import division
from __future__ import print_function
from locale import D_T_FMT

import argparse

import torch
import torch.nn as nn
from torch import optim
import os
import numpy as np   
import pickle
from sklearn.metrics import roc_auc_score
from multiprocessing import cpu_count
from utils import *
from eval import *
from GDTM import evolveLSTM, MLP


parser = argparse.ArgumentParser('G-DTM topic model')
parser.add_argument('--country_name',type=str,default='UK',help='Name of country dataset, e.g UK/US/Australia')
parser.add_argument('--num_epochs',type=int,default=50,help='Number of iterations (set to 30 as default, but 50+ is recommended.)')
parser.add_argument('--num_nodes',type=int,default=2000,help='Number of nodes (set to 2000 as default, 1000/1500/2000 can be choosed)')
parser.add_argument('--n_topic',type=int,default=60,help='Num of topics')
parser.add_argument('--lr',type=float,default=0.0002,help='Learning rate')
parser.add_argument('--alpha',type=float,default=0.001,help='weight of regularization loss(set to lower than 0.00001)')

parser.add_argument('--optimizer',type=str,default='Adam',help='The optimizer to the loss, e.g Adam/SGD')
parser.add_argument('--rnn_emb',type=int,default=300,help="The dimension of the latent topic vectors (default:300)")
parser.add_argument('--time_slice',type=int,default=4,help="The time slice of the LSTM input (default:4)")
parser.add_argument('--no_below',type=int,default=1,help="The minimum number of occurrences of the topic word(defalut:1)")
parser.add_argument('--no_above',type=float,default=0.9,help="The argest proportion of the topic word in all words(defalut:0.9)")
parser.add_argument('--param_n',type=int,default=1,help='The round of parameters')
args = parser.parse_args()


def main():
    global args
    country_name = args.country_name
    num_epochs = args.num_epochs
    num_nodes = args.num_nodes
    n_topic = args.n_topic
    lr = args.lr
    alpha = args.alpha
    optimizer = args.optimizer
    n_topic = args.n_topic
    time_slice = args.time_slice
    no_below = args.no_below
    no_above = args.no_above
    param_n = args.param_n
    rnn_emb = args.rnn_emb

    # n_cpu = cpu_count()-2 if cpu_count()>2 else 2
    # device = torch.device('cuda')

    # load the dataset
    train_data, train_binary_adj, word_info, time_steps = DocDataset2(country_name, num_nodes, time_slice)

    net = evolveLSTM(train_data.size(-1), rnn_outdim=rnn_emb, out_dim=600)
    if os.path.exists('./ckpt/GDTM-model-{}-{}'.format(country_name, num_nodes)+'.pth'):
        print("load model")
        net.load_state_dict(torch.load('./ckpt/GDTM-model-{}-{}'.format(country_name, num_nodes)+'.pth'))

    criterion = nn.MSELoss()
    if optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
    
    old_loss = np.inf
    best_loss = np.inf
    patients = 10
    bad = 0
    belta = 1000
    
    for _ in range(num_epochs):
        print("####################### epoch ", _, "###########################")
        for i in range(time_steps-time_slice+1):
            data = [train_data[i:i+time_slice-1], word_info]
            net.train()
            cell_output, output = net(data)
            sim_out = torch.mm(output, output.t()) 

            Smooth_loss = smoothness(cell_output) 
            Laplac_loss = regularization(train_binary_adj[i:i+time_slice-1], cell_output)
            MSE_loss = criterion(sim_out, train_binary_adj[i+time_slice-1]) 
            # loss = MSE_loss + alpha*Laplac_loss + belta*Smooth_loss
            loss = belta*MSE_loss + alpha*Laplac_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            nsim_out=sim_out.detach().numpy()
            ntrain_binary_adj = train_binary_adj[i+time_slice-1].detach().numpy()
            train_auc=roc_auc_score(ntrain_binary_adj, nsim_out)
            # print("steps:{}, train loss:{}, train AUC: {}".format(i, loss.item(), train_auc))
            print("steps:{}, train loss:{:.4f}, MSE_loss:{:.4f}, Laplac_loss:{:.4f}, train AUC:{:.4f}".format(i, loss.item(), MSE_loss.item(), Laplac_loss.item(), train_auc))

            if loss < old_loss:
                bad = 0
            else:
                bad += 1
            if loss < best_loss:
                state_dict = net.state_dict()
                best_loss = loss
            old_loss = loss
            if bad == patients:
                print("early stop!")
                break

    print("Best train loss:", best_loss)
    torch.save(state_dict, './ckpt/GDTM-model-{}-{}'.format(country_name, num_nodes)+'.pth')
    net.load_state_dict(state_dict)
    torch.save(net, './ckpt/GDTM-model-{}-{}'.format(country_name, num_nodes)+'.model')
    # Get the prediction result. Using set to save
    pred_sets = {}
    net.eval()
    for i in range(time_steps-time_slice+1):
        data = [train_data[i:i+time_slice-1], word_info]
        cell_output, output = net(data)
        pred_sets[3+i+time_slice-1] = output.detach().numpy()
        # pred_sets[3+i+time_trim-1] = cell_output[-1]

    # Save the result
    res_path = './data/{}/{}/topic_res-{}.pkl'.format(country_name, num_nodes, param_n)
    with open(res_path, 'wb') as file_obj:
        pickle.dump(pred_sets, file_obj)
    print("OK! Training model finish!")

    # clustering the topic and evalution
    eval(country_name, num_nodes, n_topic, no_below, no_above, param_n)





if __name__ == '__main__':
    main()
    
