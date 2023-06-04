from utils.tsne_g import draw_tsne
import utils.g_library as g_library

if __name__ == "__main__":
    dataset = g_library.construct_dataset(
        name = 'dblp4GTN',task = 'node_classification')
    draw_tsne(dataset,'./sne.png')
    print()