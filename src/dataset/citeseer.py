import json
import random

import pandas as pd
import torch
from torch.utils.data import Dataset


class CiteseerDataset(Dataset):
    def __init__(self, link_prediction=False):
        super().__init__()
        self.link_prediction = link_prediction
        self.graph = torch.load(self.processed_file_names[0])
        self.text = pd.read_csv(self.processed_file_names[1])
        self.prompt = "\nQuestion: Which of the following categories does this paper belong to: Agents, Artificial Intelligence, Database, Information Retrieval, Machine Learning, or Human Computer Interaction?\n\nAnswer: "
        self.graph_type = 'Text Attributed Graph'
        self.num_features = 3703
        self.num_classes = 6

    def has_edge(self, node1, node2):
        # Assuming edge_index is a [2, num_edges] tensor with each column being an edge
        edges = self.graph.edge_index.t().tolist()
        return [node1, node2] in edges or [node2, node1] in edges

    def __len__(self):
        """Return the len of the dataset."""
        if self.link_prediction:
            return self.graph.edge_index.shape[1] * 2
        else:
            return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            if self.link_prediction:
                num_edges = self.graph.edge_index.shape[1]
                # Adjust the total number of samples to reflect the new ratio
                total_samples = num_edges * 4  # 3 positive samples for every 1 negative sample

                if index < total_samples:
                    mod_index = index % 4
                    edge_index = index // 4

                    if mod_index < 3:  # Generate more positive samples
                        # Positive sample
                        edge = self.graph.edge_index[:, edge_index]
                        return {
                            'node1': edge[0].item(),
                            'node2': edge[1].item(),
                            'label': 1
                        }
                    else:
                        # Negative sample
                        while True:
                            edge = self.graph.edge_index[:, random.randint(0, num_edges - 1)]
                            node1 = edge[0].item()
                            node2 = random.randint(0, self.graph.num_nodes - 1)
                            if node1 != node2 and not self.has_edge(node1, node2):
                                return {
                                    'node1': node1,
                                    'node2': node2,
                                    'label': 0
                                }

                #### previous version ######
                # is_positive_sample = index % 2 == 0
                # edge_index = index // 2
                #
                # if is_positive_sample:
                #     # Positive sample
                #     edge = self.graph.edge_index[:, edge_index]
                #     # print(f"Positive sample: index={index}, node1={edge[0].item()}, node2={edge[1].item()}, label=1")
                #     return {
                #         'node1': edge[0].item(),
                #         'node2': edge[1].item(),
                #         'label': 1
                #     }
                # else:
                #     # Negative sample
                #     while True:
                #         # node1 = random.randint(0, self.graph.num_nodes - 1)
                #         # node2 = random.randint(0, self.graph.num_nodes - 1)
                #         edge = self.graph.edge_index[:, index - self.graph.edge_index.shape[1]]
                #         node1 = edge[0].item()
                #         # node1 = random.randint(0, self.graph.num_nodes - 1)
                #         node2 = random.randint(0, self.graph.num_nodes - 1)
                #         if node1 != node2 and not self.has_edge(node1, node2):
                #             # print(f"Negative sample: index={index}, node1={node1}, node2={node2}, label=0")
                #             return {
                #                 'node1': node1,
                #                 'node2': node2,
                #                 'label': 0
                #             }

            else:
                text = self.text.iloc[index]
                return {
                    'id': int(text['node_id']),
                    'label': text['label'],
                    'desc': f'Abstract: {text["abstract"]}',
                    'question': self.prompt,
                }

    @property
    def processed_file_names(self) -> str:
        return ['dataset/tape_citeseer/processed/data.pt', 'dataset/tape_citeseer/processed/text.csv']

    def get_idx_split(self):

        # Load the saved indices
        with open('dataset/tape_citeseer/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open('dataset/tape_citeseer/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open('dataset/tape_citeseer/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]
        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


if __name__ == '__main__':
    dataset = CiteseerDataset(link_prediction=True)

    print(dataset.graph)
    print(dataset.prompt)
    print(json.dumps(dataset[0], indent=4))

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')

