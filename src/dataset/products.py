import json
import torch
import pandas as pd
from torch.utils.data import Dataset


class ProductsDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = pd.read_csv(self.processed_file_names[1])
        self.text.label.fillna('NaN', inplace=True)
        
        
        self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        
        self.graph_type = 'Product co-purchasing network'
        adj = self.graph.adj_t.to_dense()
        edge_index = torch.nonzero(adj, as_tuple=False).T
        self.graph.edge_index = edge_index
        self.num_features = 100
        self.num_classes = 47

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            text = self.text.iloc[index]
            return {
                'id': index,
                'label': text['label'],
                'desc': f'Products: {text["title"]}\nDescription: {text["content"]}',
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['dataset/tape_products/processed/data.pt', 'dataset/tape_products/processed/text.csv']

    def get_idx_split(self):

        # Load the saved indices
        with open('dataset/tape_products/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open('dataset/tape_products/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open('dataset/tape_products/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]
        return {'train': train_indices, 'val': val_indices, 'test': test_indices}




if __name__ == '__main__':
    dataset = ProductsDataset()

    print(dataset.graph)
    print(dataset.prompt)
    print(json.dumps(dataset[21], indent=4))

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
