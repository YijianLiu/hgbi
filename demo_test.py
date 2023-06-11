import utils.hgbi as hgbi

if __name__ == "__main__":

    ds_node = hgbi.construct_dataset(
        name = 'ohgbl-yelp2',task = 'link_prediction')
    print(ds_node.g)

    # ds_link =  g_library.construct_dataset(
    #     name = 'amazon4SLICE',task = 'link_prediction')
    print()