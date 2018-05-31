import os
import tensorflow as tf
import numpy as np

import train_util as tu
from models import alexnet
from qutils import * 

k_patches = 1
n_layers = 8
layer_names = ['conv%d' % (l+1) for l in range(5)] + ['fc%d' % (l+1) for l in range(3)]

n_alex_nets = 5

for subj_id in range(n_alex_nets): 
    ckpt_path = '/tigress/qlu/logs/ILSVRC2012/ckpt-alexnet%.2d' % (subj_id)
    imagenet_path = '/scratch/gpfs/qlu/ILSVRC2012'
    condition = 'ILSVRC2012_img_val'
    imagenet_test_im_path = os.path.join(imagenet_path, condition)
    print(ckpt_path)

    """create log dirs"""
    # create the top dir for logging activity
    act_path = os.path.join(ckpt_path, 'acts', condition)
    if not os.path.exists(act_path):
        os.makedirs(act_path)
    print(act_path) 
    print(layer_names)

    # create a subdir for each layer
    layer_spec_dirs = []
    for layer_name in layer_names: 
        layer_spec_dir = os.path.join(act_path, layer_name)
        layer_spec_dirs.append(layer_spec_dir)
        if not os.path.exists(layer_spec_dir):
            os.makedirs(layer_spec_dir)
            print('Create dir: %s' % layer_spec_dir)

    """ get imagenet images """
    test_images = sorted(os.listdir(os.path.join(imagenet_path, 'ILSVRC2012_img_val')))
    test_labels = tu.read_test_labels(os.path.join(imagenet_path, 'data/ILSVRC2012_validation_ground_truth.txt'))
    n_imgs = len(test_images)
    print(n_imgs)


    """ build graph """
    # set up the computational graph 
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, 1000])

    _, pred = alexnet.classifier(x, 1.0)

    # load the model back 
    saver = tf.train.Saver()


    """"""
    sess = tf.InteractiveSession()
    saver.restore(sess, os.path.join(ckpt_path, 'alexnet-cnn.ckpt'))

    # get layer name 
    conv1 = tf.get_default_graph().get_tensor_by_name('alexnet_cnn/alexnet_cnn_conv1/Relu:0')
    conv2 = tf.get_default_graph().get_tensor_by_name('alexnet_cnn/alexnet_cnn_conv2/Relu:0')
    conv3 = tf.get_default_graph().get_tensor_by_name('alexnet_cnn/alexnet_cnn_conv3/Relu:0')
    conv4 = tf.get_default_graph().get_tensor_by_name('alexnet_cnn/alexnet_cnn_conv4/Relu:0')
    conv5 = tf.get_default_graph().get_tensor_by_name('alexnet_cnn/alexnet_cnn_conv5/Relu:0')
    fc1 = tf.get_default_graph().get_tensor_by_name('alexnet_classifier/alexnet_classifier_fc1/Relu:0')
    fc2 = tf.get_default_graph().get_tensor_by_name('alexnet_classifier/alexnet_classifier_fc2/Relu:0')
    fc3 = tf.get_default_graph().get_tensor_by_name('alexnet_classifier/alexnet_classifier_output/Add:0')


    """ fetch activity """
    for im_idx in range(n_imgs): 
        n_imgs_batch = 1
        img_name = 'img_%.8d' % (im_idx+1)
        print(img_name)

        cur_image = tu.read_resized_image(
            os.path.join(imagenet_test_im_path, test_images[im_idx]))

        # gather neural activity 
        fetch_list = [conv1, conv2, conv3, conv4, conv5, fc1, fc2, fc3]
        fetched_list = sess.run(fetch_list, feed_dict={x: cur_image})
        [conv1_f, conv2_f, conv3_f, conv4_f, conv5_f, fc1_f, fc2_f, fc3_f] = fetched_list
        # end of loop

        # flatten arrays for conv layers 
        conv1_f = np.reshape(conv1_f, [n_imgs_batch, k_patches, -1])
        conv2_f = np.reshape(conv2_f, [n_imgs_batch, k_patches, -1])
        conv3_f = np.reshape(conv3_f, [n_imgs_batch, k_patches, -1])
        conv4_f = np.reshape(conv4_f, [n_imgs_batch, k_patches, -1])
        conv5_f = np.reshape(conv5_f, [n_imgs_batch, k_patches, -1])

        layer_acts_fs = [conv1_f, conv2_f, conv3_f, conv4_f, conv5_f, fc1_f, fc2_f, fc3_f]
        for layer_act, layer_spec_dir in zip(layer_acts_fs, layer_spec_dirs): 
            np.savez_compressed(os.path.join(layer_spec_dir, img_name), acts=layer_act)
