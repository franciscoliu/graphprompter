import os
import pandas as pd
import numpy as np


from src.dataset.preprocess.generate_split import generate_split
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch


def get_cora_casestudy():
    data_X, data_Y, data_citeid, data_edges = parse_cora()

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    return data, data_citeid

# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun


def parse_cora():
    path = 'dataset/cora_orig/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora():
    data, data_citeid = get_cora_casestudy()

    with open('dataset/cora_orig/mccallum/cora/papers')as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = 'dataset/cora_orig/mccallum/cora/extractions/'
    text = []
    for pid in data_citeid:
        fn = pid_filename[pid]
        with open(path+fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            if 'Title:' in line:
                ti = line.split('Title:')[1].strip()
            if 'Abstract:' in line:
                ab = line.split('Abstract:')[1].strip()
        text.append({'title': ti, 'abstract': ab})

    return data, text


def preprocess():
    classes = ['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Method', 'Reinforcement Learning', 'Rule Learning', 'Theory']
    data, text = get_raw_text_cora()
    labels = [classes[y] for y in data.y]

    # save graph data
    path = 'dataset/tape_cora/processed'
    os.makedirs(path, exist_ok=True)
    torch.save(data, f'{path}/data.pt')

    # save text data
    df = pd.DataFrame(text)
    df['node_id'] = np.arange(data.num_nodes)
    df['label'] = labels
    df.to_csv(f'{path}/text.csv', index=False, columns=['node_id', 'label', 'title', 'abstract'])

    # save split
    generate_split(data.num_nodes, 'dataset/tape_cora/split')


if __name__ == '__main__':
    print("Preprocessing tape_cora dataset...")
    preprocess()
    print("Done!")
