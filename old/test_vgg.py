from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import functools
import random
import collections
import math
import time
import scipy.ndimage as sn 
import scipy.io as sio
from vgg import *

import warnings
from functools import partial
from nets import resnet_utils
from models import *
import ops

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--class_num", type=int, default=1000)
parser.add_argument("--num_examples", type=int, default=1000)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--input_type", default="tfrecord", choices=["tfrecord", "image", "directory"])

parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--target_size", type=int, default=224, help="scale images to this size before cropping to 256x256")

parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")

a = parser.parse_args()

EPS = 1e-12

Examples = collections.namedtuple("Examples", "images, labels, steps_per_epoch")
Model = collections.namedtuple("Model", "logits,onehot_labels,accuracy, label_indices, loss, grads_and_vars, train")

def convert(image):
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

def transform(image, seed):
    """ Transform image to augment data.
    Including:
        flip: flip image horizontally
        monochrome: rgb to grayscale
        center_crop: center crop image to make sure weight == height
        random_crop: resize image to a larger scale_size and randomly crop it to target a.target_size.
        resize: resize image to [a.scale_size, a.scale_size]        
    """
    # make a copy of image, otherwise get wrong results for unkwon reason
    r = image
    height = r.get_shape()[0] # h, w, c
    width = r.get_shape()[1]
    print(height, 'x x x', width)
    print(r.get_shape())
    if a.flip:
        r = tf.image.random_flip_left_right(r, seed=seed)
    if not 1: #height == width:
        # center crop to correct ratio
        print("height not equal to width")
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        print(ow, "oooooo", oh)
        r = tf.image.crop_to_bounding_box(image=r, offset_height=oh, offset_width=ow, target_height=size, target_width=size)
    
    # resize to a.target_size
    r = tf.image.resize_images(r, [a.target_size, a.target_size], method=tf.image.ResizeMethod.AREA)

    return r 

def parse_function(example_proto):
    '''
        
    '''            
    features = {
                'height':  tf.FixedLenFeature([], tf.int64),
                'width':  tf.FixedLenFeature([], tf.int64),
                'depth':  tf.FixedLenFeature([], tf.int64),
                'image':  tf.FixedLenFeature([], tf.string),    
                'label':  tf.FixedLenFeature([], tf.int64),
            }        
    
    parsed_features = tf.parse_single_example(example_proto, features=features) 
    
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    height = tf.cast(parsed_features['height'], tf.int32)
    width = tf.cast(parsed_features['width'], tf.int32)
    label = parsed_features['label']
    image = tf.reshape(image, [height, width, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.float64)
    image = image * 2. -1.    

    seed = random.randint(0, 2**31 - 1)
    image = transform(image, seed) 

    return image, label

def parse_function_cifar10(example_proto):
    '''

    '''            
    features = {
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/format': tf.FixedLenFeature([], tf.string),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
            }        
    
    parsed_features = tf.parse_single_example(example_proto, features=features) 
    
    #image = tf.decode_raw(parsed_features['image/encoded'], tf.uint8)
    image = tf.image.decode_png(parsed_features['image/encoded'], tf.uint8)
    height = tf.cast(parsed_features['image/height'], tf.int32)
    width = tf.cast(parsed_features['image/width'], tf.int32)
    label = parsed_features['image/class/label']
    image = tf.reshape(image, [height, width, 4])
    image = tf.image.convert_image_dtype(image[:,:,:3], dtype=tf.float32)
    image = image * 2. -1.    

    seed = random.randint(0, 2**31 - 1)
    image = transform(image, seed) 

    return image, label

def read_tfrecord():
    tfrecord_fn = glob.glob(os.path.join(a.input_dir, "*.tfrecords"))
    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    #dataset = dataset.map(parse_function)  # Parse the record into tensors. 
    dataset = dataset.map(parse_function_cifar10)  # Parse the record into tensors. 
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    #dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(a.batch_size)
    iterator = dataset.make_one_shot_iterator()

    image, label = iterator.get_next()
    image.set_shape([a.batch_size, a.target_size, a.target_size, 3])

    steps_per_epoch = int(math.ceil(a.num_examples / a.batch_size))
    
    # show read results for code test
    #sess = tf.Session()
    #image, label, path = sess.run(iterator.get_next())
    #Image.fromarray(image[0,:,:,:], "RGB").save('./a1.jpg')
    #print(label, path)
    
    return Examples(
        #paths=None,
        images=image,
        labels=label,
        steps_per_epoch=steps_per_epoch
    ), iterator
    

def create_model(images, label_indices, class_num):
    onehot_labels = tf.one_hot(label_indices, class_num)

    with tf.name_scope("vgg_network"):
        logits, endpoints = vgg_16(images, num_classes=class_num,global_pool=True)
    with tf.name_scope("softmax_cross_entropy_loss"):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    with tf.name_scope("accuracy"):
        accuracy = tf.metrics.accuracy(labels=label_indices, predictions=tf.argmax(logits, axis=1))
    
    with tf.name_scope("training_network"):
        tvars = [var for var in tf.trainable_variables()]
        optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        grads_and_vars = optim.compute_gradients(loss, var_list=tvars, colocate_gradients_with_ops=True)
        train_op = optim.apply_gradients(grads_and_vars)

        #with tf.control_dependencies([train_op]):
        #    ema = tf.train.ExponentialMovingAverage(decay=0.99)
        #    update_losses = ema.apply([loss])
        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        logits=logits,
        onehot_labels=onehot_labels,
        accuracy=accuracy,
        label_indices=label_indices,
        loss=loss,
        grads_and_vars=grads_and_vars,
        train=tf.group(train_op, incr_global_step),
        )

def main():
    ################ check directory #########################################
    if not os.path.exists(a.output_dir):    
        os.makedirs(a.output_dir)

    if a.checkpoint is None:
        raise Exception("checkpoint required for test mode")

    ############### display/save setting #####################################
    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    ################ read TFRecord dataset ###################################
    input_batch, iterator  = read_tfrecord()

    ################ creat model  ############################################
    # inputs and targets are of size [batch_size, height, width, channels]
    model = create_model(input_batch.images, input_batch.labels, a.class_num)

    ################ summaries ###############################################
    # undo colorization splitting on images that we use for display/output
    images = deprocess(input_batch.images)

    # YuhangLi: only save a part of images for saving driver space.
    with tf.name_scope("encode_images"):
        display_fetches = {
            #"paths": input_batch.paths,
            "images": tf.map_fn(tf.image.encode_png, convert(images), dtype=tf.string, name="images"),
            "label": input_batch.labels
        }

    
    ################ configuration ##############################################
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    f = open("log.txt",mode='wb')

    ############### session ######################################################
    with sv.managed_session(config=sess_config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = input_batch.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        ######################## test ############################
        if a.mode == "test":
            # at most, process the test data once
            max_steps = min(input_batch.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results, step=step)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                # temporaly commented, error for unknown reason    
                # index_path = append_index(filesets)

            print("wrote index at", index_path)
    return

main()