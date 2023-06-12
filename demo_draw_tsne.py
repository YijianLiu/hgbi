from utils.tsne_g import draw_tsne
import utils.hgbi as hgbi

if __name__ == "__main__":
    dataset = hgbi.build_dataset(
        name = 'dblp4GTN',task = 'node_classification')
    draw_tsne(dataset,'./sne.png')
    print()