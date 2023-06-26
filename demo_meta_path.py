from utils.meta_path_analyse import number_meta_path
import utils.hgbi as hgbi
from openhgnn.dataset import IMDB4GTNDataset, DBLP4GTNDataset
if __name__ == "__main__":
    dataset = hgbi.build_dataset(
        name = 'dblp4GTN',task = 'node_classification')
    g = dataset[0]
    meta_path_nums, heterophily, edge_radio  = number_meta_path(g, meta_paths_dict=dataset.meta_paths_dict, strength=2)
    print(meta_path_nums)
    print(heterophily)
    print(edge_radio)
