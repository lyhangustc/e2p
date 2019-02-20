import argparse
import os
import json
import glob
import functools
import random
import collections
import math
import time
from PIL import Image
import numpy as np

import tensorflow as tf
from vgg import *

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", default="train", choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--class_num", type=int, required=True)
parser.add_argument("--num_examples", type=int, required=True)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--target_size", type=int, default=224, help="scale images to this size before cropping to 256x256")

parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=1000, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--evaluate_freq", type=int, default=5000, help="evaluate training data every save_freq steps, 0 to disable")

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
    if a.flip:
        r = tf.image.random_flip_left_right(r, seed=seed)
    if not 1: #height == width:
        # center crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
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
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = image * 2. -1.    

    seed = random.randint(0, 2**31 - 1)
    image = transform(image, seed) 

    return image, label

def read_tfrecord():
    tfrecord_fn = glob.glob(os.path.join(a.input_dir, "*.tfrecords"))
    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    dataset = dataset.map(parse_function)  # Parse the record into tensors. 
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    #dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(a.batch_size)
    iterator = dataset.make_one_shot_iterator()

    image, label = iterator.get_next()
    image.set_shape([a.batch_size, a.target_size, a.target_size, 3])

    steps_per_epoch = int(math.ceil(a.num_examples / a.batch_size))
    
    # show read results for code test
    sess = tf.Session()
    image_np, label_np = sess.run(iterator.get_next())
    image_np = np.ceil((image_np[0,:,:,:] + 1.) / 2.)
    Image.fromarray(image_np, "RGB").save('./11.jpg')
    print(label_np)
    
    return Examples(
        #paths=None,
        images=image,
        labels=label,
        steps_per_epoch=steps_per_epoch
    ), iterator
    

def create_model(images, label_indices, class_num=1000):
    onehot_labels = tf.one_hot(label_indices, class_num)
    with tf.name_scope("vgg_network"):
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py
        logits, endpoints = vgg_16(images, num_classes=class_num,global_pool=True)
    with tf.name_scope("softmax_cross_entropy_loss"):
        #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_indices, logits=logits))
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    with tf.name_scope("accuracy"):
        accuracy = tf.metrics.accuracy(labels=label_indices, predictions=tf.argmax(logits, axis=1))
    
    with tf.name_scope("training_network"):
        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)
        optimize = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)



    return Model(
        logits=logits,
        onehot_labels=onehot_labels,
        accuracy=accuracy,
        label_indices=label_indices,
        #loss=ema.average(loss),
        loss=loss,
        grads_and_vars=None,
        train=tf.group(optimize, incr_global_step),
        )

def main():
    ################ check directory #########################################
    if not os.path.exists(a.output_dir):    
        os.makedirs(a.output_dir)
    ################ read TFRecord dataset ###################################
    input_batch, iterator  = read_tfrecord()

    ################ creat model  ############################################
    model = create_model(input_batch.images, input_batch.labels, a.class_num)

    ################ configuration ###########################################
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        max_steps = 2**32
        start = time.time()
        fetches = {
                    "train": model.train,
                }
        fetches["label_indices"] = model.label_indices
        fetches["loss"] = model.loss
        #fetches["accuracy"] = model.accuracy
        for step in range(max_steps):
            results = sess.run(fetches)

            if step % 49 == 0:
                print("loss", results["loss"])

main()