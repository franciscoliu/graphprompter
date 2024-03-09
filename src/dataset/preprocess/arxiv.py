import os
import torch
import pandas as pd
import numpy as np

from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T


def get_raw_text_arxiv():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]

    nodeidx2paperid = pd.read_csv('dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')
    raw_text = pd.read_csv('dataset/ogbn_arxiv_orig/titleabs.tsv', sep='\t', header=None, names=['paper id', 'title', 'abs'])
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        text.append({'title': ti.strip(), 'abstract': ab.strip()})

    # use public split provided by OGB
    idx_split = dataset.get_idx_split()

    train_indices = idx_split['train'].tolist()
    val_indices = idx_split['valid'].tolist()
    test_indices = idx_split['test'].tolist()
    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    path = 'dataset/tape_arxiv/split'
    os.makedirs(path, exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))

    return data, text


def preprocess():

    data, text = get_raw_text_arxiv()

    label2text = pd.read_csv("./dataset/ogbn_arxiv_orig/label2text.csv")
    # cs.XX, eg., cs.AI
    classes = label2text['arxiv category'].apply(lambda x: 'cs.'+x.split(' ')[-1]).tolist()
    labels = [classes[y.item()] for y in data.y]

    # eg., Artificial Intelligence
    classes = label2text['answer'].tolist()
    full_labels = [classes[y.item()] for y in data.y]

    # save graph data
    path = 'dataset/tape_arxiv/processed'
    os.makedirs(path, exist_ok=True)
    torch.save(data, f'{path}/data.pt')

    # save text data
    df = pd.DataFrame(text)
    df['node_id'] = np.arange(data.num_nodes)
    df['label'] = labels
    df['full_label'] = full_labels
    df.to_csv(f'{path}/text.csv', index=False, columns=['node_id', 'label', 'full_label', 'title', 'abstract'])


if __name__ == '__main__':
    print("Preprocessing tape_arxiv dataset...")
    preprocess()
    print("Done!")
