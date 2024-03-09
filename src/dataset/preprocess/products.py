import os
import torch
import numpy as np
import pandas as pd


def get_raw_text_products():
    data = torch.load('dataset/ogbn_products/ogbn-products_subset.pt')
    text = pd.read_csv('dataset/ogbn_products_orig/ogbn-products_subset.csv')

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
    data, text = get_raw_text_products()
    labelidx2productcategory = pd.read_csv('dataset/ogbn_products/mapping/labelidx2productcategory.csv.gz', compression='gzip')
    classes = labelidx2productcategory['product category'].tolist()
    classes[24] = 'None'
    classes[45] = 'Furniture & Decor'

    labels = [classes[y.item()] for y in data.y]
    path = 'dataset/tape_products/processed'

    df = pd.DataFrame(text)
    df['node_id'] = np.arange(data.num_nodes)
    df['label'] = labels

    df.to_csv(f'{path}/text.csv', index=False, columns=['nid', 'label', 'title', 'content'])


if __name__ == '__main__':
    process()
