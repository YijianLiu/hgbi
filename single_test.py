import g_library

ds_node = g_library.construct_dataset(name = 'acm4NSHE',task = 'node_classification')
print()
'''
ds_node contains dataset details, e.g. in_dim, meta_paths, category, num_classes
ds_node.g is Graph on DGL format
'''

ds_link =  g_library.construct_dataset(name = 'HGBl-LastFM',task = 'link_prediction')
print()
'''
ds_link contains dataset details, e.g. target_link, target_link_r, node_type,
ds_link.g is Graph on DGL format
'''

