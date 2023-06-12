import utils.hgbi as hgbi

if __name__ == "__main__":

    #Build Risk Product Detection Dataset
    ds_node = hgbi.build_dataset(
        name = 'RPDD',task = 'node_classification')
    '''
    ds_node contains dataset details:
    e.g. in_dim, meta_paths, category, num_classes
    ds_node.g is Graph on DGL format
    '''

    #Build Takeout Recommendation Dataset
    ds_link =  hgbi.build_dataset(
        name = 'TRD',task = 'link_prediction')
    '''
    ds_link contains dataset details:
    e.g. target_link, target_link_r, node_type,
    ds_link.g is Graph on DGL format
    '''

    #construct own dataset
    #rebuild
    ds = hgbi.MyDataset(
        name="my_graph",path="./graph.bin")
    ds_link = hgbi.AsLinkPredictionDataset(
        ds, target_link=['user-buy-poi'],
        target_link_r=['rev_user-buy-poi'],
        split_ratio=[0.5, 0.3, 0.3],
        neg_ratio=3,
        neg_sampler='global'
    )
