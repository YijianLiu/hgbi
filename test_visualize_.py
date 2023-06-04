from unittest import TestCase
from utils import *
import utils.g_library as g_library

if __name__ == "__main__":

    dataset = g_library.construct_dataset(name = 'ICDM', task = 'node_classification')
    g = dataset.g
    print(dataset.meta_paths_dict)
    plot_number_metapath(g, meta_paths_dict=dataset.meta_paths_dict, save_path='./plot_number_metapath.png')

    dataset = g_library.construct_dataset(name = 'MTWM', task = 'link_prediction')
    g = dataset.g
    print(dataset.meta_paths_dict)
    plot_number_metapath(g, meta_paths_dict=dataset.meta_paths_dict, save_path='./plot_number_metapath.png')