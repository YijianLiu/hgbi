import utils.hgbi as hgbi

ds_node = hgbi.build_dataset(
    name = 'acm4NSHE',task = 'node_classification')
print(ds_node.g)

ds_link = hgbi.build_dataset(
    name = 'ohgbl-yelp2',task = 'link_prediction')
print(ds_link.g)
