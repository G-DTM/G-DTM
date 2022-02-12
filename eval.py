from operator import pos
from platform import node
from networkx.classes.function import degree
from networkx.classes.graph import Graph

import os
import numpy as np
import math
import pickle
from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import ast
from utils import *

def select_vecotr(vecset, month_file, keywords, no_below, no_above):
    vec2set = []
    month = os.listdir(month_file)
    doc_len = len(month)
    for mfile in month:
        use_file = month_file + mfile
        fi = open(use_file, 'r')
        data = fi.read()
        fi.close()
        vec1set = []
        for word in keywords.values():
            if data.find(word) == -1:
                vec1set.append(0)
            else:
                vec1set.append(1)
        vec2set.append(np.array(vec1set))
    vec2set = np.sum(np.array(vec2set),axis=0)
    idx_set, neat_keywords = [], {}
    j = 0
    for i in range(len(vec2set)):
        if vec2set[i] >= no_below and vec2set[i]/len(keywords) <= no_above:
            idx_set.append(i)
            neat_keywords[j] = keywords[i]
            j = j+1
    neat_vecset = vecset[idx_set]
    return neat_vecset, neat_keywords, idx_set


def clustering(vec_data, country_name, num_nodes, n_clusters, slice, no_below, no_above, methods_idx=1):
    month_file = './data/{}/new/{}/'.format(country_name, slice)
    tmp = open('./data/{}/{}/keywords-nodedict.pkl'.format(country_name, num_nodes), 'rb')
    keywords = ast.literal_eval(pickle.load(tmp))

    neat_vector, neat_keywords, idx_set = select_vecotr(vec_data, month_file, keywords, no_below, no_above)
    cluter_methods = {1:'KMeans', 2:'Spectral', 3:'Hierarchical'}
    name = cluter_methods[methods_idx]
    if methods_idx == 1:
        # Kmeans
        k_means = KMeans(n_clusters=n_clusters, random_state=50, init='k-means++')
        k_means.fit(neat_vector)
        pred_y = k_means.predict(neat_vector)
    elif methods_idx == 2:
        return 1
    elif methods_idx == 3:
        return 1

    total = []
    for i in range(n_clusters):
        total.append([])
    for i in range(len(neat_keywords)):
        num = pred_y[i]
        total[num].append(neat_keywords[i])
    prob3set = []
    for i in range(n_clusters):
        prob2set = mutual_info(total[i], month_file)
        prob3set.append(prob2set)
    prob3tc = np.sum(np.array(prob3set))/n_clusters
    print('Month {} Topic Coherence: {:.4f}'.format(slice,prob3tc))
    return prob3tc, prob3set, total

def GraphTopicModel(time_info, word_info, node_info, country_name, num_nodes, n_topic, slice, no_below, no_above):

    # all_features = np.concatenate(((word_info+node_info), time_info), axis=1)
    all_features = time_info + word_info   
    # alpha = 0.1
    # tmp_info = node_info + alpha*word_info    
    # tmp_info = np.concatenate((node_info, time_info), axis=1)
    # all_features =  np.matmul(tmp_info, np.transpose(tmp_info))
    # estimator01 = PCA(n_components=300)
    # word_info = estimator01.fit_transform(word_info)
    # word_info_pca = estimator01.fit_transform(word_info)
    # alpha = 0.2
    # all_features = cosine_similarity(node_info + alpha*word_info)
    # estimator02 = PCA(n_components=200) 
    # all_features = estimator02.fit_transform(all_features)
    prob3tc, prob_set, clustering_res = clustering(all_features, country_name, num_nodes, n_topic, slice, no_below, no_above, methods_idx=1)
    return prob3tc,prob_set, clustering_res


def eval(country_name, num_nodes, n_topic, no_below, no_above, param_n):

    eval_root_path = './data/{}/{}/topic_res-{}.pkl'.format(country_name,num_nodes,param_n)
    word_info = np.array(load_data('./data/{}/{}/word2vec.pkl'.format(country_name, num_nodes)))
    node_info = np.array(load_data('./data/{}/{}/node2vec.pkl'.format(country_name, num_nodes)))
    with open(eval_root_path, 'rb') as file_obj:
        res = pickle.load(file=file_obj) 

    # Cluster the topics
    all_tc_sets = []
    all_cluster_res = []
    for month, time_info in res.items(): 
        prob3tc, prob3set, clustering_res = GraphTopicModel(time_info, word_info, node_info[month-3], country_name, num_nodes, n_topic, month, no_below, no_above)
        all_tc_sets.append(prob3tc)
        all_cluster_res.append(clustering_res)
    print("Average of All tc value: ", sum(all_tc_sets)/len(all_tc_sets))
    

