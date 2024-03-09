# adapted from https://github.com/jcatw/scnn
import os
import torch
import numpy as np
import pandas as pd

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from src.dataset.preprocess.generate_split import generate_split

# return citeseer dataset as pytorch geometric Data object together with 60/20/20 split, and list of citeseer IDs


def get_citeseer_casestudy():
    data_X, data_Y, data_citeid, data_edges = parse_citeseer()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    # load data
    data_name = 'Citeseer'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    return data, data_citeid


def parse_citeseer():
    path = 'dataset/citeseer_orig/citeseer'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(float)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(
        ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'])}
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


def get_raw_text_citeseer():
    data, data_citeid = get_citeseer_casestudy()

    with open('dataset/citeseer_orig/citeseer_texts.txt') as f:
        lines = f.read().splitlines()

    paper_ids = [lines[i] for i in range(len(lines)) if i % 3 == 0]
    abstracts = [lines[i] for i in range(len(lines)) if i % 3 == 1]
    # labels = [lines[i] for i in range(len(lines)) if i % 3 == 2]

    df = pd.DataFrame([paper_ids, abstracts]).transpose()
    df.columns = ['paper_id', 'abstract']

    df2 = pd.DataFrame([np.arange(data.num_nodes), data_citeid]).transpose()
    df2.columns = ['node_id', 'paper_id']

    # intersection of two dataframes
    df3 = pd.merge(df, df2, how='outer', on=['paper_id'])
    df3.sort_values(by=['node_id'], inplace=True)
    df3.fillna('None', inplace=True)

    return data, df3['abstract'].tolist()


def preprocess():

    data, text = get_raw_text_citeseer()

    # classes = ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI']
    classes = ['Agents', 'Artificial Intelligence', 'Database', 'Information Retrieval', 'Machine Learning', 'Human Computer Interaction']
    labels = [classes[y] for y in data.y]

    # save graph data
    path = 'dataset/tape_citeseer/processed'
    os.makedirs(path, exist_ok=True)
    torch.save(data, f'{path}/data.pt')

    # save text data
    df = pd.DataFrame(text)
    df['node_id'] = np.arange(data.num_nodes)
    df['label'] = labels
    df['abstract'] = text
    df.to_csv(f'{path}/text.csv', index=False, columns=['node_id', 'label', 'abstract'])

    # save split
    generate_split(data.num_nodes, 'dataset/tape_citeseer/split')


if __name__ == '__main__':
    print("Preprocessing tape_citeseer dataset...")
    preprocess()
    print("Done!")
