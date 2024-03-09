import os
import torch
import numpy as np
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset


def get_raw_text_products_full():

    trn = pd.read_json(
        'dataset/ogbn_products/Amazon-3M.raw/trn.json', lines=True)
    tst = pd.read_json(
        'dataset/ogbn_products/Amazon-3M.raw/tst.json', lines=True)
    raw = pd.concat([trn, tst], axis=0)

    # get title & content
    nodeidx2asin = pd.read_csv(
        'dataset/ogbn_products/mapping/nodeidx2asin.csv.gz', compression='gzip')
    df = pd.merge(raw, nodeidx2asin, left_on='uid', right_on='asin')
    df.sort_values(by=['node idx'], inplace=True)

    # get label
    dataset = PygNodePropPredDataset(name='ogbn-products')
    data = dataset[0]
    labelidx2productcategory = pd.read_csv(
        'dataset/ogbn_products/mapping/labelidx2productcategory.csv.gz', compression='gzip')
    classes = labelidx2productcategory['product category'].tolist()
    classes[24] = 'None'
    classes[45] = 'Furniture & Decor'
    labels = [classes[y.item()] for y in data.y]
    df['label'] = labels

    # save to csv
    df.to_csv('dataset/tape_products/raw.csv', index=False,
              columns=['node idx', 'asin', 'title', 'content', 'label'])

    # use public split provided by OGB
    idx_split = dataset.get_idx_split()

    train_indices = idx_split['train'].tolist()
    val_indices = idx_split['valid'].tolist()
    test_indices = idx_split['test'].tolist()
    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    path = 'dataset/tape_products/split'
    os.makedirs(path, exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))

    return data, df


def get_raw_text_products():
    data = torch.load('dataset/ogbn_products/ogbn-products_subset.pt')
    text = pd.read_csv('dataset/ogbn_products_orig/ogbn-products_subset.csv')

    labelidx2productcategory = pd.read_csv(
        'dataset/ogbn_products/mapping/labelidx2productcategory.csv.gz', compression='gzip')
    classes = labelidx2productcategory['product category'].tolist()
    classes[24] = 'None'
    classes[45] = 'Furniture & Decor'

    labels = [classes[y.item()] for y in data.y]
    path = 'dataset/tape_products/processed'

    df = pd.DataFrame(text)
    df['node_id'] = np.arange(data.num_nodes)
    df['label'] = labels

    df.to_csv(f'{path}/text.csv', index=False,
              columns=['nid', 'label', 'title', 'content'])

    data.edge_index = data.adj_t.to_symmetric()

    train_indices = [i for i, mask in enumerate(data.train_mask) if mask]
    val_indices = [i for i, mask in enumerate(data.val_mask) if mask]
    test_indices = [i for i, mask in enumerate(data.test_mask) if mask]

    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    path = 'dataset/tape_products/split'
    os.makedirs(path, exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))
    return data, text


def process():
    get_raw_text_products_full()


if __name__ == '__main__':
    process()
