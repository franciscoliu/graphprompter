import torch
from torch_geometric.utils import mask_to_index, index_to_mask


def batch_subgraph(edge_index,
                   node_ids,
                   num_nodes,
                   num_hops=3,
                   fans_out=(50, 50, 50)):
    # print(f"Starting batch_subgraph for node_ids: {node_ids}")
    subset_list, edge_index_sub_list, mapping_list, batch_list = [], [], [], []

    row, col = edge_index
    inc_num = 0
    batch_id = 0

    for node_idx in node_ids:
        # print(f"Processing node_idx: {node_idx}")
        subsets = [node_idx.flatten()]
        node_mask = row.new_empty(num_nodes, dtype=torch.bool)

        for hop in range(num_hops):
            node_mask.fill_(False)

            node_mask[subsets[-1]] = True
            edge_mask = torch.index_select(node_mask, 0, row)

            neighbors = col[edge_mask]
            # print(f"Hop {hop}, neighbors before fan-out limit: {neighbors}")

            if len(neighbors) > fans_out[hop]:
                perm = torch.randperm(len(neighbors))[:fans_out[hop]]
                neighbors = neighbors[perm]
            # print(f"Hop {hop}, neighbors after fan-out limit: {neighbors}")


            subsets.append(neighbors)

        subset, ind = torch.unique(torch.cat(subsets), return_inverse=True)
        # print(f"Final subset for node_idx {node_idx}: {subset}")


        node_mask = index_to_mask(subset, size=num_nodes)
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index_sub = edge_index[:, edge_mask]

        # Relabel Node
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long, device=edge_index.device)
        node_idx[subset] = torch.arange(node_mask.sum().item(), device=edge_index.device)
        edge_index_sub = node_idx[edge_index_sub]

        # Batching Graph
        edge_index_sub += inc_num

        subset_list.append(subset)
        edge_index_sub_list.append(edge_index_sub)
        mapping_list.append(inc_num + ind[0].item())
        batch_list.extend([batch_id for _ in range(len(subset))])

        inc_num += len(subset)
        batch_id += 1

    subset = torch.cat(subset_list)
    # print(f"Total subset after processing node_ids {node_ids}: {subset}")

    mapping = torch.as_tensor(mapping_list)
    batch = torch.as_tensor(batch_list)
    edge_index_sub = torch.cat(edge_index_sub_list, dim=1)

    return subset, edge_index_sub, mapping, batch


class TAGCollator(object):
    def __init__(self, graph, link_prediction=False, max_nodes_per_hop=50):
        self.graph = graph
        self.link_prediction = link_prediction
        self.max_nodes_per_hop = max_nodes_per_hop  # Maximum expected neighborhood size

    def pad_features(self, features, max_size):
        """Pad the features tensor to the given max size."""
        pad_size = max_size - features.size(0)
        if pad_size > 0:
            padding = torch.zeros(pad_size, features.size(1), device=features.device)
            features = torch.cat([features, padding], dim=0)
        return features

    def __call__(self, original_batch):
        if self.link_prediction:
            mybatch = {}
            for k in original_batch[0].keys():
                mybatch[k] = [d[k] for d in original_batch]

            # Extract node lists and labels
            # node1_list = [item['node1'] for item in original_batch]
            # node2_list = [item['node2'] for item in original_batch]
            labels = torch.tensor([item['label'] for item in original_batch])

            # Process subsets for node1 and node2
            subset_1, edge_index_sub_1, mapping_1, batch_1 = batch_subgraph(
                edge_index=self.graph.edge_index,
                node_ids=torch.tensor(mybatch['node1']),
                num_nodes=self.graph.num_nodes
            )

            subset_2, edge_index_sub_2, mapping_2, batch_2 = batch_subgraph(
                edge_index=self.graph.edge_index,
                node_ids=torch.tensor(mybatch['node2']),
                num_nodes=self.graph.num_nodes
            )

            # Fetch features for node1 and node2
            node1_features = self.graph.x[subset_1]
            node2_features = self.graph.x[subset_2]

            # Pad features to ensure equal size
            max_nodes = max(node1_features.size(0), node2_features.size(0), self.max_nodes_per_hop)
            node1_features = self.pad_features(node1_features, max_nodes)
            node2_features = self.pad_features(node2_features, max_nodes)

            # Create a dictionary for the batch
            batch = {
                'node1': mybatch['node1'],
                'node2': mybatch['node2'],
                'node1_features': node1_features,
                'node2_features': node2_features,
                'label': labels,
                'edge_index_1': edge_index_sub_1,
                'edge_index_2': edge_index_sub_2,
                'mapping_1': mapping_1,
                'mapping_2': mapping_2,
                'batch_1': batch_1,
                'batch_2': batch_2,
            }
            return batch


        else:
            mybatch = {}
            for k in original_batch[0].keys():
                mybatch[k] = [d[k] for d in original_batch]

            subset, edge_index_sub, mapping, batch = batch_subgraph(
                edge_index=self.graph.edge_index,
                node_ids=torch.tensor(mybatch['id']),
                num_nodes=self.graph.num_nodes
            )

            mybatch['x'] = self.graph.x[subset]
            mybatch['y'] = self.graph.y[subset]
            mybatch['edge_index'] = edge_index_sub
            mybatch['mapping'] = mapping
            mybatch['batch'] = batch

            return mybatch


collate_funcs = {
    'cora': TAGCollator,
    'citeseer': TAGCollator,
    'pubmed': TAGCollator,
    'arxiv': TAGCollator,
    'products': TAGCollator,
}
