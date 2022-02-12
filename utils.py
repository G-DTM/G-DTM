import torch
import pickle
import numpy as np
import os
import ast
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

def load_data(path):
    with open(path, 'rb') as file_obj:
        data = pickle.load(file_obj)
    return data

def train_test_split(data, test_start):
    train_data = data[:-2, :, :]
    test_data = data[test_start:-1, :, :]
    return torch.Tensor(train_data), torch.Tensor(test_data)

## smoothness loss
def smoothness(cells_output):
    smoothness_loss = 0
    for i in range(len(cells_output)-1):
        smoothness_loss += ((cells_output[i] - cells_output[i+1]).norm())**2
    return smoothness_loss

def regularization(adj, signal):
    loss_reg = 0
    for i in range(len(adj)):
        degree = torch.sum(adj[i], axis=1)
        degree = torch.diag(degree)
        Laplac = degree - adj[i]
        # loss_reg += torch.trace(torch.transpose(signal[i], 0, 1)@Laplac@signal[i])
        loss_reg += torch.trace((signal[i].T)@Laplac@signal[i])
    return loss_reg

def mutual_info(wordset, month_file):
    vec2set = []
    month = os.listdir(month_file)
    doc_len = len(month)

    for mfile in month:
        use_file = month_file + mfile
        fi = open(use_file, 'r')
        data = fi.read()
        fi.close()
        vec1set = []
        for word in wordset:
            if data.find(word) == -1:
                vec1set.append(0)
            else:
                vec1set.append(1)
        vec2set.append(np.array(vec1set))
    vec2set = np.array(vec2set)
    # freq_one = cal_one_freq(vec2set)
    prob2set = []
    for i in range(vec2set.shape[1]):
        prob1set = []
        for j in range(vec2set.shape[1]):
            if i == j:
                prob1set.append(1)
                continue
            dw_i, dw_j = cal_one_freq(i, vec2set), cal_one_freq(j, vec2set)
            dw_ij = cal_two_freq(i, j, vec2set)
            prob = cal_mutual_info(dw_i, dw_j, dw_ij, doc_len)
            prob1set.append(prob)
        prob2set.append(np.array(prob1set))
    prob2set = np.array(prob2set)
    return  np.sum(prob2set)/(prob2set.shape[0]*prob2set.shape[1])

def cal_one_freq(idx, vec_set):
    one_freq = np.sum(vec_set, axis=0)
    return one_freq[idx]

def cal_two_freq(idx1, idx2, vec2set):
    column1 = vec2set[:, idx1]
    column2 = vec2set[:, idx2]
    two_freq = list((column1 + column2)).count(2)
    return two_freq

def cal_mutual_info(dw1, dw2, dw12, D):
    if dw1 == 0 or dw2== 0 or dw12 == 0:
        return -1 + ((np.log((dw1 + 1) / D) + np.log((dw2 + 1) / D)) / np.log((dw12 + 1) / D))
    return -1 + ((np.log((dw1) / D) + np.log((dw2) / D)) / np.log((dw12) / D))

def cluster(vec_data, country_name, num_nodes, n_clusters, slice, methods_idx=1):
    month_file = './data/{}/new/{}/'.format(country_name, slice)
    tmp = open('./data/{}/{}/keywords-nodedict.pkl'.format(country_name, num_nodes), 'rb')
    keywords = ast.literal_eval(pickle.load(tmp))
    # cluter_methods = {1:'KMeans', 2:'Spectral', 3:'Hierarchical'}
    # name = cluter_methods[methods_idx]

    if methods_idx == 1:
        # Kmeans
        k_means = KMeans(n_clusters=n_clusters, random_state=50, init='k-means++')
        k_means.fit(vec_data)
        pred_y = k_means.predict(vec_data)
        # print("Calinski-Harabasz Score ", "n_cluster=", n_clusters, "chscore=",
        #     calinski_harabasz_score(vec_data, pred_y), 'scscore=', silhouette_score(vec_data, pred_y), 'dbi:',
        #     metrics.davies_bouldin_score(vec_data, pred_y))
    elif methods_idx == 2:
        # Spectral; 
        return 1
    elif methods_idx == 3:
        # Hierarchical; 
        return 1

    total = []
    for i in range(n_clusters):
        total.append([])
    for i in range(len(keywords)):
        num = pred_y[i]
        total[num].append(keywords[i])
    prob3set = []
    for i in range(n_clusters):
        prob2set = mutual_info(total[i], month_file)
        prob3set.append(prob2set)
    prob3tc = np.sum(np.array(prob3set))/n_clusters
    print('Month {} Topic Coherence: {:.4f}'.format(slice,prob3tc))
    return prob3tc, prob3set, total
    
# Forcasting task dataset format
def DocDataset1(country_name, num_nodes, time_slice):

    node_data = load_data('./data/{}/{}/node2vec.pkl'.format(country_name, num_nodes))
    vec_data = torch.Tensor(load_data('./data/{}/{}/word2vec.pkl'.format(country_name, num_nodes)))
    
    # Index of the start time of the dataset
    time_begin = 3
    # Time slice length of dataset
    time_steps = node_data.shape[0]  
    test_begin = - time_slice
    train_data, test_data = train_test_split(node_data[:,:,:], test_start=test_begin)

    adj_path = './data/{}/{}/'.format(country_name, num_nodes)
    
    train_binary_adj = []
    for i in range(time_begin, time_steps+time_begin):
        binary_adj = torch.load(os.path.join(adj_path, 'adj-unnorm-2-{}.pth'.format(int(i)))).cpu().type(torch.FloatTensor)
        train_binary_adj.append(binary_adj)

    test_binary_adj = torch.load(os.path.join(adj_path, 'adj-unnorm-2-{}.pth'.format(int(time_steps+time_begin-1)))).cpu()
    return train_data, test_data, train_binary_adj, test_binary_adj, vec_data, time_steps


# Dynamic model task dataset format
def DocDataset2(country_name, num_nodes, time_slice):
    node_data = load_data('./data/{}/{}/node2vec.pkl'.format(country_name, num_nodes))
    vec_data = torch.Tensor(load_data('./data/{}/{}/word2vec.pkl'.format(country_name, num_nodes)))
    # Index of the start time of the dataset
    time_begin = 3
    # Time slice length of dataset
    time_steps = node_data.shape[0] 
    train_data = torch.Tensor(node_data)
    adj_path = './data/{}/{}/'.format(country_name, num_nodes)
    
    train_binary_adj = []
    for i in range(time_begin, time_steps+time_begin):
        binary_adj = torch.load(os.path.join(adj_path, 'adj-unnorm-2-{}.pth'.format(int(i)))).cpu().type(torch.FloatTensor)
        train_binary_adj.append(binary_adj)
        
    return train_data, train_binary_adj, vec_data, time_steps
