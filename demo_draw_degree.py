from utils.tsne_g import draw_tsne
import utils.hgbi as hgbi
from utils import *

if __name__ == "__main__":
    dataset = hgbi.build_dataset(
        name = 'dblp4GTN',task = 'node_classification')
    plot_degree_dist(dataset.g,'./degree.png')