from utils.meta_path_analyse import number_meta_path
import utils.g_library as g_library
from openhgnn.dataset import IMDB4GTNDataset, DBLP4GTNDataset
if __name__ == "__main__":
    dataset = g_library.construct_dataset(
        name = 'dblp4GTN',task = 'node_classification')
    g = dataset[0]
    meta_path_nums, connectivity_strength, homogeneity = number_meta_path(g, meta_paths_dict=dataset.meta_paths_dict, strength=2)
    print(meta_path_nums)
    print(connectivity_strength)
    print(homogeneity)