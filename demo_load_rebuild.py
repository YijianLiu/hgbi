import utils.g_library as g_library

if __name__ == "__main__":
    ds_node = g_library.construct_dataset(
        name = 'ICDM',task = 'node_classification')
    '''
    ds_node contains dataset details:
    e.g. in_dim, meta_paths, category, num_classes
    ds_node.g is Graph on DGL format
    '''
    ds_link =  g_library.construct_dataset(
        name = 'MTWN',task = 'link_prediction')
    '''
    ds_link contains dataset details:
    e.g. target_link, target_link_r, node_type,
    ds_link.g is Graph on DGL format
    '''

    #construct own dataset
    #rebuild
    ds = g_library.MyDataset(
        name="my_graph",path="./graph.bin")
    ds_link = g_library.AsLinkPredictionDataset(
        ds, target_link=['user-buy-poi'],
        target_link_r=['rev_user-buy-poi'],
        split_ratio=[0.5, 0.3, 0.3],
        neg_ratio=3,
        neg_sampler='global'
    )
