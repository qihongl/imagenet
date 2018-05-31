import numpy as np 
import os 
import glob 

def read_stimuli_order(stimuli_order_fname, MAPS_ROOT):
    """ load the stimuli ordering, constructed by "construct_stimuli_order.py"
    """
    stimuli_order_fpath = os.path.join(MAPS_ROOT, stimuli_order_fname)
    stimuli_order = np.load(stimuli_order_fpath)
    stimuli_order_full_ids = stimuli_order['full_ids']
    stimuli_order_im_paths = stimuli_order['image_paths']
    
    return stimuli_order_full_ids, stimuli_order_im_paths


def construct_stimuli_ordering(paths_wnid):
    # compute the number of folders
    num_classes = len(paths_wnid)
    # preallocate
    full_ids = []
    im_paths = []
    for i in range(num_classes):
        paths_wnid[i]
        # get the path to all images
        paths_image = glob.glob(paths_wnid[i] + '*.JPEG')
        num_images = len(paths_image)
        for j in range(num_images):
            path_image = paths_image[j]
            fname_image = os.path.basename(path_image)
            # print(fname_image)
            full_ids.append(dict_im_full_name_to_full_id[fname_image])
            im_paths.append(path_image)
    return full_ids, im_paths