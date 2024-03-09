from src.dataset.cora import CoraDataset
from src.dataset.citeseer import CiteseerDataset
from src.dataset.pubmed import PubmedDataset
from src.dataset.arxiv import ArxivDataset
from src.dataset.products import ProductsDataset


load_dataset = {
    'cora': CoraDataset,
    'citeseer': CiteseerDataset,
    'pubmed': PubmedDataset,
    'arxiv': ArxivDataset,
    'products': ProductsDataset,
}
